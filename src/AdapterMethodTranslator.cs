using System.Reflection;
using System.Reflection.Emit;

namespace ParallelGpu;

/// <summary>Maps built-in memory views and registered integer indexers to contiguous GPU descriptors.</summary>
internal sealed class AdapterMethodTranslator(TypeBuilder helpers)
{
    private readonly Dictionary<MethodBase, MethodInfo> methods = new();

    internal bool TryEmit(ILGenerator il, MethodInfo method)
    {
        var declaring = method.DeclaringType!;
        Type? element = GpuCollectionAdapters.ElementType(declaring);
        bool extension = declaring == typeof(MemoryExtensions) && method.IsGenericMethod &&
            method.Name is "AsSpan" or "AsMemory";
        if (extension)
        {
            element = method.GetGenericArguments()[0];
            if (method.GetParameters().Length == 0 ||
                CapturedCollection.ElementType(method.GetParameters()[0].ParameterType) is null)
                return false;
        }
        if (element is null)
            return false;
        if ((extension && method.GetParameters().Skip(1).Any(parameter => parameter.ParameterType != typeof(int))) ||
            (method.Name == "Slice" && method.GetParameters().Any(parameter => parameter.ParameterType != typeof(int))))
            throw new NotSupportedException("GPU memory slicing requires integer start and length arguments; Index and Range overloads are unsupported.");
        if (!methods.TryGetValue(method, out var replacement))
            replacement = Translate(method, element, extension);
        il.Emit(OpCodes.Call, replacement);
        return true;
    }

    internal bool TryEmitConstructor(ILGenerator il, OpCode code, ConstructorInfo constructor)
    {
        var type = constructor.DeclaringType!;
        if (!type.IsGenericType || !IsValueView(type))
            return false;
        if (constructor.GetParameters().Length == 0 ||
            constructor.GetParameters().Any(parameter => parameter.ParameterType != typeof(int) &&
                CapturedCollection.ElementType(parameter.ParameterType) is null))
            throw new NotSupportedException($"Memory constructor '{constructor}' cannot be translated to contiguous GPU storage.");
        if (!methods.TryGetValue(constructor, out var replacement))
        {
            var parameters = constructor.GetParameters().Select(parameter => ScalarMethodTranslator.MapType(parameter.ParameterType)).ToArray();
            replacement = helpers.DefineMethod("ViewConstructor" + methods.Count,
                MethodAttributes.Public | MethodAttributes.Static, typeof(DeviceCollection), parameters);
            methods.Add(constructor, replacement);
            var body = ((MethodBuilder)replacement).GetILGenerator();
            body.Emit(OpCodes.Ldarg_0);
            EmitSlice(body, parameters.Length - 1, 1);
            if (IsReadOnly(type))
                body.Emit(OpCodes.Call, SliceMethod(nameof(DeviceCollectionSlice.ReadOnly)));
            else
                body.Emit(OpCodes.Call, SliceMethod(nameof(DeviceCollectionSlice.Normalize)));
            body.Emit(OpCodes.Ret);
        }
        if (code == OpCodes.Newobj)
            il.Emit(OpCodes.Call, replacement);
        else
        {
            var arguments = constructor.GetParameters().Select(parameter => il.DeclareLocal(ScalarMethodTranslator.MapType(parameter.ParameterType))).ToArray();
            for (int index = arguments.Length - 1; index >= 0; index--)
                il.Emit(OpCodes.Stloc, arguments[index]);
            foreach (var argument in arguments)
                il.Emit(OpCodes.Ldloc, argument);
            il.Emit(OpCodes.Call, replacement);
            il.Emit(OpCodes.Stobj, typeof(DeviceCollection));
        }
        return true;
    }

    private MethodInfo Translate(MethodInfo method, Type element, bool extension)
    {
        var declaring = method.DeclaringType!;
        bool valueReceiver = !method.IsStatic && declaring.IsValueType;
        var parameters = method.GetParameters().Select(parameter => ScalarMethodTranslator.MapType(parameter.ParameterType)).ToList();
        if (!method.IsStatic)
            parameters.Insert(0, valueReceiver ? typeof(DeviceCollection).MakeByRefType() : typeof(DeviceCollection));
        var result = ScalarMethodTranslator.MapType(method.ReturnType);
        var builder = helpers.DefineMethod("ViewMethod" + methods.Count,
            MethodAttributes.Public | MethodAttributes.Static, result, parameters.ToArray());
        methods.Add(method, builder);
        var il = builder.GetILGenerator();
        il.Emit(OpCodes.Ldarg_0);
        if (valueReceiver)
            il.Emit(OpCodes.Ldobj, typeof(DeviceCollection));
        if (extension)
        {
            EmitSlice(il, parameters.Count - 1, 1);
            il.Emit(OpCodes.Call, SliceMethod(IsReadOnly(method.ReturnType)
                ? nameof(DeviceCollectionSlice.ReadOnly) : nameof(DeviceCollectionSlice.Normalize)));
        }
        else if (method.Name is "get_Count" or "get_Length")
            il.Emit(OpCodes.Call, valueReceiver ? SliceMethod(nameof(DeviceCollectionSlice.Count))
                : typeof(DeviceCollection).GetMethod(nameof(DeviceCollection.Count))!);
        else if (method.Name == "get_IsEmpty")
            il.Emit(OpCodes.Call, SliceMethod(nameof(DeviceCollectionSlice.IsEmpty)));
        else if (method.Name == "get_Item")
        {
            if (method.GetParameters().Length != 1 || method.GetParameters()[0].ParameterType != typeof(int))
                throw new NotSupportedException("GPU collection adapters require an integer indexer.");
            if (valueReceiver)
                il.Emit(OpCodes.Call, SliceMethod(nameof(DeviceCollectionSlice.Normalize)));
            il.Emit(OpCodes.Ldarg_1);
            il.Emit(OpCodes.Call, typeof(DeviceCollection).GetMethod(method.ReturnType.IsByRef
                ? IsReadOnly(declaring) ? nameof(DeviceCollection.ReadOnlyAddress) : nameof(DeviceCollection.Address)
                : nameof(DeviceCollection.Read))!.MakeGenericMethod(DeviceValueType.MapType(element)));
        }
        else if (method.Name == "set_Item")
        {
            if (method.GetParameters().Length != 2 || method.GetParameters()[0].ParameterType != typeof(int))
                throw new NotSupportedException("GPU collection adapters require an integer indexer.");
            if (valueReceiver)
                il.Emit(OpCodes.Call, SliceMethod(nameof(DeviceCollectionSlice.Normalize)));
            il.Emit(OpCodes.Ldarg_1);
            il.Emit(OpCodes.Ldarg_2);
            il.Emit(OpCodes.Call, typeof(DeviceCollection).GetMethod(nameof(DeviceCollection.Write))!
                .MakeGenericMethod(DeviceValueType.MapType(element)));
        }
        else if (IsValueView(declaring) && method.Name is "Slice" or "get_Span" or "op_Implicit" or "op_Explicit")
        {
            int argumentCount = parameters.Count - 1;
            if (method.Name == "Slice")
                EmitSlice(il, argumentCount, 1);
            else if (argumentCount != 0)
                throw new NotSupportedException($"Memory operation '{method}' is unsupported inside GPU loops.");
            il.Emit(OpCodes.Call, SliceMethod(IsReadOnly(method.ReturnType)
                ? nameof(DeviceCollectionSlice.ReadOnly) : nameof(DeviceCollectionSlice.Normalize)));
        }
        else
            throw new NotSupportedException($"Collection adapter operation '{declaring.FullName}.{method.Name}' is unsupported inside GPU loops. Use its integer indexer, length, or contiguous slice.");
        il.Emit(OpCodes.Ret);
        return builder;
    }

    private static void EmitSlice(ILGenerator il, int arguments, int startArgument)
    {
        if (arguments == 0)
            return;
        if (arguments is not (1 or 2))
            throw new NotSupportedException("GPU memory slicing supports an integer start and optional integer length.");
        il.Emit(OpCodes.Ldarg, (short)startArgument);
        if (arguments == 2)
            il.Emit(OpCodes.Ldarg, (short)(startArgument + 1));
        il.Emit(OpCodes.Call, SliceMethod(arguments == 1
            ? nameof(DeviceCollectionSlice.SliceFrom) : nameof(DeviceCollectionSlice.Slice)));
    }

    private static MethodInfo SliceMethod(string name) => typeof(DeviceCollectionSlice).GetMethod(name)!;

    private static bool IsValueView(Type type) => type.IsGenericType &&
        (type.GetGenericTypeDefinition() == typeof(ArraySegment<>) || type.GetGenericTypeDefinition() == typeof(Memory<>) ||
         type.GetGenericTypeDefinition() == typeof(ReadOnlyMemory<>) || type.GetGenericTypeDefinition() == typeof(Span<>) ||
         type.GetGenericTypeDefinition() == typeof(ReadOnlySpan<>));

    private static bool IsReadOnly(Type type) => type.IsGenericType &&
        (type.GetGenericTypeDefinition() == typeof(ReadOnlyMemory<>) || type.GetGenericTypeDefinition() == typeof(ReadOnlySpan<>));
}
