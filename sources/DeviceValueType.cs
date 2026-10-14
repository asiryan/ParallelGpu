using System.Collections.Concurrent;
using System.Reflection;
using System.Reflection.Emit;

namespace ParallelGpu;

/// <summary>Recognizes and maps fixed-size device values without managed references.</summary>
/// <remarks>Visible structures with sequential layout, default packing, and unchanged field types retain their native
/// representation. Other supported structures use public mirrors preserving all instance fields, with UInt16
/// character fields and Int32 Boolean fields.</remarks>
internal static class DeviceValueType
{
    /// <summary>Cached structural checks shared by compilation and collection registration.</summary>
    private static readonly ConcurrentDictionary<Type, bool> supported = new();
    /// <summary>Recognizes values whose implicit array storage cannot be represented by reflected fields.</summary>
    private static readonly ConcurrentDictionary<Type, bool> hiddenStorage = new();
    /// <summary>Host structures mapped to their device mirrors and field correspondences.</summary>
    private static readonly Dictionary<Type, (Type Type, Dictionary<FieldInfo, FieldInfo> Fields)> mappings = new();
    /// <summary>Serializes creation of mirrors shared by different executors.</summary>
    private static readonly object sync = new();
    /// <summary>The assembly containing publicly accessible device mirrors and host converters.</summary>
    private static readonly AssemblyBuilder assembly = AssemblyBuilder.DefineDynamicAssembly(
        new AssemblyName("ParallelGpu.Values"), AssemblyBuilderAccess.Run);
    /// <summary>The module shared by persistent device mirrors and their host converters.</summary>
    private static readonly ModuleBuilder module = assembly.DefineDynamicModule("Values");
    /// <summary>Assemblies granting converter access to private structure fields.</summary>
    private static readonly HashSet<string> accessAssemblies = new();
    /// <summary>Provides distinct generated converter type names.</summary>
    private static int converterCount;
    /// <summary>Boxed converters for structure values captured in loop closures.</summary>
    private static readonly ConcurrentDictionary<Type, Func<object?, object>> converters = new();

    /// <summary>Checks primitives, enums, and recursively composed structures without managed references.</summary>
    /// <param name="type">The candidate value type.</param>
    /// <returns>True for a supported primitive, enum, or structure whose instance fields use supported device values.</returns>
    /// <remarks>Explicit layouts, inline arrays, fixed buffers, byref-like values, and native pointers are unsupported.
    /// Nullable values use a mapped presence flag and payload.
    /// Method compatibility is checked separately by the device compiler.</remarks>
    internal static bool IsSupported(Type type) => supported.GetOrAdd(type, candidate =>
    {
        if (candidate.IsEnum)
            return IsSupported(Enum.GetUnderlyingType(candidate));
        if (candidate.IsPrimitive)
            return candidate != typeof(IntPtr) && candidate != typeof(UIntPtr);
        return candidate != typeof(void) && candidate.IsValueType && !candidate.IsByRefLike &&
            !candidate.IsExplicitLayout && !HasHiddenStorage(candidate) &&
            Fields(candidate).All(field => IsSupported(field.FieldType));
    });

    /// <summary>Rejects inline arrays and fixed buffers before allocation, conversion, or host callbacks.</summary>
    /// <param name="type">The original value type, including nested structures.</param>
    internal static void ValidateStorage(Type type)
    {
        if (HasHiddenStorage(type))
            throw new NotSupportedException($"Inline arrays and fixed buffers are not supported GPU value representations: '{type}'.");
    }

    /// <summary>Checks metadata without instantiating attributes, including attributes from newer runtimes.</summary>
    private static bool HasHiddenStorage(Type type) => hiddenStorage.GetOrAdd(type, candidate =>
        candidate.IsValueType && !candidate.IsPrimitive && !candidate.IsEnum &&
        (candidate.CustomAttributes.Any(attribute => attribute.AttributeType.FullName is
            "System.Runtime.CompilerServices.InlineArrayAttribute" or
            "System.Runtime.CompilerServices.UnsafeValueTypeAttribute") ||
        Fields(candidate).Any(field => field.CustomAttributes.Any(attribute => attribute.AttributeType.FullName ==
            "System.Runtime.CompilerServices.FixedBufferAttribute") || HasHiddenStorage(field.FieldType))));

    /// <summary>Gets instance fields in their metadata order, including private and readonly fields.</summary>
    /// <param name="type">The structure whose fields are inspected.</param>
    /// <returns>The ordered instance fields.</returns>
    private static FieldInfo[] Fields(Type type) => type
        .GetFields(BindingFlags.Instance | BindingFlags.Public | BindingFlags.NonPublic)
        .OrderBy(field => field.MetadataToken).ToArray();

    /// <summary>Maps characters, by-reference value types, and structures needing an accessible device mirror.</summary>
    /// <param name="type">The original value or reference type.</param>
    /// <returns>The mapped device value type, or the original type when no supported mapping applies.</returns>
    internal static Type MapType(Type type)
    {
        if (type.IsByRef)
            return MapType(type.GetElementType()!).MakeByRefType();
        ValidateStorage(type);
        if (type == typeof(char))
            return typeof(ushort);
        if (!IsSupported(type) || type.IsPrimitive || type.IsEnum)
            return type;
        lock (sync)
        {
            if (mappings.TryGetValue(type, out var existing))
                return existing.Type;
            var fields = Fields(type);
            var types = fields.Select(field => field.FieldType == typeof(bool)
                ? typeof(int) : MapType(field.FieldType)).ToArray();
            if (type.IsVisible && !type.IsAutoLayout && (type.StructLayoutAttribute?.Pack ?? 0) == 0 &&
                fields.Select(field => field.FieldType).SequenceEqual(types))
            {
                mappings.Add(type, (type, fields.ToDictionary(field => field, field => field)));
                return type;
            }
            var builder = module.DefineType("Value" + mappings.Count,
                TypeAttributes.Public | TypeAttributes.Sealed | TypeAttributes.SequentialLayout, typeof(ValueType));
            for (int i = 0; i < fields.Length; i++)
                builder.DefineField(fields[i].Name, types[i], FieldAttributes.Public);
            var mapped = builder.CreateType()!;
            mappings.Add(type, (mapped, fields.ToDictionary(field => field, field => mapped.GetField(field.Name)!)));
            return mapped;
        }
    }

    /// <summary>Maps an instance field to the corresponding field on a device mirror.</summary>
    /// <param name="field">The original instance field.</param>
    /// <returns>The original field or its mapped counterpart.</returns>
    internal static FieldInfo MapField(FieldInfo field)
    {
        if (field.IsStatic || MapType(field.DeclaringType!) == field.DeclaringType)
            return field;
        lock (sync)
            return mappings[field.DeclaringType!].Fields[field];
    }

    /// <summary>Converts a captured value into its device representation.</summary>
    /// <param name="value">The boxed host value.</param>
    /// <param name="type">The original host value type.</param>
    /// <returns>A boxed value suitable for the generated capture field.</returns>
    internal static object ToDevice(object? value, Type type) => MapType(type) == type ? value! :
        converters.GetOrAdd(type, source =>
        {
            lock (sync)
            {
                var (builder, method) = DefineConverter(source, typeof(object), typeof(object));
                var il = method.GetILGenerator();
                il.Emit(OpCodes.Ldarg_0);
                il.Emit(OpCodes.Unbox_Any, source);
                EmitConversion(il, source, MapType(source));
                il.Emit(OpCodes.Box, MapType(source));
                il.Emit(OpCodes.Ret);
                return (Func<object?, object>)builder.CreateType()!.GetMethod(method.Name)!
                    .CreateDelegate(typeof(Func<object?, object>));
            }
        })(value);

    /// <summary>Builds a typed host converter used to pack or unpack value arrays.</summary>
    /// <typeparam name="TSource">The original value representation.</typeparam>
    /// <typeparam name="TTarget">The requested value representation.</typeparam>
    /// <returns>A converter that copies and converts all instance fields.</returns>
    internal static Func<TSource, TTarget> Converter<TSource, TTarget>()
    {
        lock (sync)
        {
            var (builder, method) = DefineConverter(typeof(TSource), typeof(TSource), typeof(TTarget));
            AllowAccess(typeof(TTarget), new HashSet<Type>());
            var il = method.GetILGenerator();
            il.Emit(OpCodes.Ldarg_0);
            EmitConversion(il, typeof(TSource), typeof(TTarget));
            il.Emit(OpCodes.Ret);
            return (Func<TSource, TTarget>)builder.CreateType()!.GetMethod(method.Name)!
                .CreateDelegate(typeof(Func<TSource, TTarget>));
        }
    }

    /// <summary>Defines a persistent converter next to the persistent device mirrors.</summary>
    /// <param name="original">The host type whose private fields are accessed.</param>
    /// <param name="source">The method's parameter type.</param>
    /// <param name="target">The method's return type.</param>
    /// <returns>The converter type builder and its static method.</returns>
    /// <remarks>Ordinary generated methods avoid DynamicMethod resolver finalization while mirrors remain cached.</remarks>
    private static (TypeBuilder Type, MethodBuilder Method) DefineConverter(Type original, Type source, Type target)
    {
        AllowAccess(original, new HashSet<Type>());
        var builder = module.DefineType("Converter" + converterCount++,
            TypeAttributes.Public | TypeAttributes.Abstract | TypeAttributes.Sealed);
        var method = builder.DefineMethod("Convert", MethodAttributes.Public | MethodAttributes.Static,
            target, [source]);
        return (builder, method);
    }

    /// <summary>Grants generated host converters access to private fields throughout a structure.</summary>
    /// <param name="type">The value type or generic argument to inspect.</param>
    /// <param name="visited">The types already inspected.</param>
    private static void AllowAccess(Type type, HashSet<Type> visited)
    {
        if (!visited.Add(type))
            return;
        string name = type.Assembly.GetName().Name!;
        if (accessAssemblies.Add(name))
        {
            var attribute = typeof(System.Runtime.CompilerServices.IgnoresAccessChecksToAttribute)
                .GetConstructor([typeof(string)])!;
            assembly.SetCustomAttribute(new CustomAttributeBuilder(attribute, [name]));
        }
        foreach (var argument in type.GetGenericArguments())
            AllowAccess(argument, visited);
        if (type.IsValueType && !type.IsPrimitive && !type.IsEnum)
            foreach (var field in Fields(type))
                AllowAccess(field.FieldType, visited);
    }

    /// <summary>Converts one value already on the evaluation stack, recursively copying structure fields.</summary>
    /// <param name="il">The host converter's instruction generator.</param>
    /// <param name="source">The source representation.</param>
    /// <param name="target">The target representation.</param>
    private static void EmitConversion(ILGenerator il, Type source, Type target)
    {
        if (source == target)
            return;
        if (source == typeof(bool) && target == typeof(int))
            return; // Both use an Int32 evaluation-stack value.
        if (source == typeof(int) && target == typeof(bool))
        {
            il.Emit(OpCodes.Ldc_I4_0);
            il.Emit(OpCodes.Ceq);
            il.Emit(OpCodes.Ldc_I4_0);
            il.Emit(OpCodes.Ceq);
            return;
        }
        if (source == typeof(char) || target == typeof(char))
        {
            il.Emit(OpCodes.Conv_U2);
            return;
        }
        var input = il.DeclareLocal(source);
        var output = il.DeclareLocal(target);
        il.Emit(OpCodes.Stloc, input);
        il.Emit(OpCodes.Ldloca, output);
        il.Emit(OpCodes.Initobj, target);
        bool toDevice = MapType(source) == target;
        var original = toDevice ? source : target;
        MapType(original);
        Dictionary<FieldInfo, FieldInfo> fields;
        lock (sync)
            fields = mappings[original].Fields;
        foreach (var pair in fields)
        {
            var from = toDevice ? pair.Key : pair.Value;
            var to = toDevice ? pair.Value : pair.Key;
            il.Emit(OpCodes.Ldloca, output);
            il.Emit(OpCodes.Ldloca, input);
            il.Emit(OpCodes.Ldfld, from);
            EmitConversion(il, from.FieldType, to.FieldType);
            il.Emit(OpCodes.Stfld, to);
        }
        il.Emit(OpCodes.Ldloc, output);
    }
}
