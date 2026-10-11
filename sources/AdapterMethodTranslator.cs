using System.Collections.Concurrent;
using System.Reflection;
using System.Reflection.Emit;

namespace ParallelGpu;

/// <summary>Maps built-in memory views and registered integer indexers to contiguous GPU descriptors.</summary>
internal sealed class AdapterMethodTranslator(TypeBuilder helpers)
{
    private readonly Dictionary<(MethodBase Method, Type? Receiver), MethodInfo> methods = new();
    private static readonly ConcurrentDictionary<ConstructorInfo, bool> nullSegmentConstructors = new();

    internal bool TryEmit(ILGenerator il, MethodInfo method, Type? receiverType = null)
    {
        var declaring = method.DeclaringType!;
        if (receiverType?.IsByRef == true)
            receiverType = receiverType.GetElementType();
        var collectionType = receiverType is not null &&
            GpuCollectionAdapters.ElementType(receiverType) is not null ? receiverType : declaring;
        Type? element = GpuCollectionAdapters.ElementType(collectionType);
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
        if (!methods.TryGetValue((method, collectionType), out var replacement))
            replacement = Translate(method, element, extension, collectionType);
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
        if (!methods.TryGetValue((constructor, null), out var replacement))
        {
            var parameters = constructor.GetParameters().Select(parameter => ScalarMethodTranslator.MapType(parameter.ParameterType)).ToArray();
            replacement = helpers.DefineMethod("ViewConstructor" + methods.Count,
                MethodAttributes.Public | MethodAttributes.Static, typeof(DeviceCollection), parameters);
            methods.Add((constructor, null), replacement);
            var body = ((MethodBuilder)replacement).GetILGenerator();
            body.Emit(OpCodes.Ldarg_0);
            if (IsSegment(type))
            {
                if (parameters.Length == 1)
                    body.Emit(OpCodes.Call, SliceMethod(nullSegmentConstructors.GetOrAdd(constructor, AllowsNullSegment)
                        ? nameof(DeviceCollectionSlice.Segment) : nameof(DeviceCollectionSlice.SegmentConstructorWhole)));
                else if (parameters.Length == 3)
                {
                    body.Emit(OpCodes.Ldarg_1);
                    body.Emit(OpCodes.Ldarg_2);
                    body.Emit(OpCodes.Call, SliceMethod(nameof(DeviceCollectionSlice.SegmentConstructor)));
                }
                else
                    throw new NotSupportedException($"Array segment constructor '{constructor}' is unsupported inside GPU loops.");
            }
            else
            {
                EmitSlice(body, parameters.Length - 1, 1);
                body.Emit(OpCodes.Call, SliceMethod(IsReadOnly(type)
                    ? nameof(DeviceCollectionSlice.ReadOnly) : nameof(DeviceCollectionSlice.Normalize)));
            }
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

    private MethodInfo Translate(MethodInfo method, Type element, bool extension, Type collectionType)
    {
        var declaring = method.DeclaringType!;
        var operation = GpuCollectionAdapters.AccessorOperation(collectionType, method, element);
        if (operation is null && !extension && !IsValueView(collectionType))
            throw new NotSupportedException($"Collection adapter operation '{declaring.FullName}.{method.Name}' is unsupported inside GPU loops. Use its integer indexer or length.");
        operation ??= method.Name;
        bool valueReceiver = !method.IsStatic && declaring.IsValueType;
        var parameters = method.GetParameters().Select(parameter => ScalarMethodTranslator.MapType(parameter.ParameterType)).ToList();
        if (!method.IsStatic)
            parameters.Insert(0, valueReceiver ? typeof(DeviceCollection).MakeByRefType() : typeof(DeviceCollection));
        var result = ScalarMethodTranslator.MapType(method.ReturnType);
        var builder = helpers.DefineMethod("ViewMethod" + methods.Count,
            MethodAttributes.Public | MethodAttributes.Static, result, parameters.ToArray());
        methods.Add((method, collectionType), builder);
        var il = builder.GetILGenerator();
        il.Emit(OpCodes.Ldarg_0);
        if (valueReceiver)
            il.Emit(OpCodes.Ldobj, typeof(DeviceCollection));
        if (extension)
        {
            EmitSlice(il, parameters.Count - 1, 1);
            il.Emit(OpCodes.Call, SliceMethod(IsReadOnly(method.ReturnType)
                ? nameof(DeviceCollectionSlice.ReadOnly) : IsSegment(method.ReturnType)
                    ? nameof(DeviceCollectionSlice.Segment) : nameof(DeviceCollectionSlice.Normalize)));
        }
        else if (operation is "get_Count" or "get_Length")
            il.Emit(OpCodes.Call, valueReceiver ? SliceMethod(nameof(DeviceCollectionSlice.Count))
                : typeof(DeviceCollection).GetMethod(nameof(DeviceCollection.Count))!);
        else if (operation == "get_IsEmpty")
            il.Emit(OpCodes.Call, SliceMethod(nameof(DeviceCollectionSlice.IsEmpty)));
        else if (operation == "get_Item")
        {
            if (method.GetParameters().Length != 1 || method.GetParameters()[0].ParameterType != typeof(int))
                throw new NotSupportedException("GPU collection adapters require an integer indexer.");
            if (valueReceiver)
                il.Emit(OpCodes.Call, SliceMethod(IsSegment(declaring)
                    ? nameof(DeviceCollectionSlice.Segment) : nameof(DeviceCollectionSlice.Normalize)));
            il.Emit(OpCodes.Ldarg_1);
            il.Emit(OpCodes.Call, typeof(DeviceCollection).GetMethod(method.ReturnType.IsByRef
                ? IsReadOnly(declaring) ? nameof(DeviceCollection.ReadOnlyAddress) : nameof(DeviceCollection.Address)
                : nameof(DeviceCollection.Read))!.MakeGenericMethod(DeviceValueType.MapType(element)));
        }
        else if (operation == "set_Item")
        {
            if (method.GetParameters().Length != 2 || method.GetParameters()[0].ParameterType != typeof(int))
                throw new NotSupportedException("GPU collection adapters require an integer indexer.");
            if (valueReceiver)
                il.Emit(OpCodes.Call, SliceMethod(IsSegment(declaring)
                    ? nameof(DeviceCollectionSlice.Segment) : nameof(DeviceCollectionSlice.Normalize)));
            il.Emit(OpCodes.Ldarg_1);
            il.Emit(OpCodes.Ldarg_2);
            il.Emit(OpCodes.Call, typeof(DeviceCollection).GetMethod(nameof(DeviceCollection.Write))!
                .MakeGenericMethod(DeviceValueType.MapType(element)));
        }
        else if (IsValueView(declaring) && method.Name is "Slice" or "get_Span" or "op_Implicit" or "op_Explicit")
        {
            int argumentCount = parameters.Count - 1;
            if (method.Name == "Slice")
                EmitSlice(il, argumentCount, 1, IsSegment(declaring));
            else if (argumentCount != 0)
                throw new NotSupportedException($"Memory operation '{method}' is unsupported inside GPU loops.");
            il.Emit(OpCodes.Call, SliceMethod(IsReadOnly(method.ReturnType)
                ? nameof(DeviceCollectionSlice.ReadOnly) : IsSegment(method.ReturnType)
                    ? nameof(DeviceCollectionSlice.Segment) : nameof(DeviceCollectionSlice.Normalize)));
        }
        else
            throw new NotSupportedException($"Collection adapter operation '{declaring.FullName}.{method.Name}' is unsupported inside GPU loops. Use its integer indexer, length, or contiguous slice.");
        il.Emit(OpCodes.Ret);
        return builder;
    }

    private static void EmitSlice(ILGenerator il, int arguments, int startArgument, bool segment = false)
    {
        if (arguments == 0)
            return;
        if (arguments is not (1 or 2))
            throw new NotSupportedException("GPU memory slicing supports an integer start and optional integer length.");
        il.Emit(OpCodes.Ldarg, (short)startArgument);
        if (arguments == 2)
            il.Emit(OpCodes.Ldarg, (short)(startArgument + 1));
        il.Emit(OpCodes.Call, SliceMethod(segment
            ? arguments == 1 ? nameof(DeviceCollectionSlice.SegmentSliceFrom) : nameof(DeviceCollectionSlice.SegmentSlice)
            : arguments == 1 ? nameof(DeviceCollectionSlice.SliceFrom) : nameof(DeviceCollectionSlice.Slice)));
    }

    /// <summary>Matches the current CLR's null-array constructor semantics during host compilation.</summary>
    private static bool AllowsNullSegment(ConstructorInfo constructor)
    {
        try
        {
            constructor.Invoke(new object?[] { null });
            return true;
        }
        catch (TargetInvocationException exception) when (exception.InnerException is ArgumentNullException)
        {
            return false;
        }
    }

    private static MethodInfo SliceMethod(string name) => typeof(DeviceCollectionSlice).GetMethod(name)!;

    private static bool IsValueView(Type type) => type.IsGenericType &&
        (type.GetGenericTypeDefinition() == typeof(ArraySegment<>) || type.GetGenericTypeDefinition() == typeof(Memory<>) ||
         type.GetGenericTypeDefinition() == typeof(ReadOnlyMemory<>) || type.GetGenericTypeDefinition() == typeof(Span<>) ||
         type.GetGenericTypeDefinition() == typeof(ReadOnlySpan<>));

    private static bool IsSegment(Type type) =>
        type.IsGenericType && type.GetGenericTypeDefinition() == typeof(ArraySegment<>);

    private static bool IsReadOnly(Type type) => type.IsGenericType &&
        (type.GetGenericTypeDefinition() == typeof(ReadOnlyMemory<>) || type.GetGenericTypeDefinition() == typeof(ReadOnlySpan<>));
}
