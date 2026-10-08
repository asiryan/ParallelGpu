using System.Reflection;
using System.Reflection.Emit;

namespace ParallelGpu;

/// <summary>Builds rank-specialized collection accessors without temporary index arrays on the device.</summary>
internal sealed class CollectionMethodTranslator(TypeBuilder helpers)
{
    private readonly Dictionary<MethodInfo, MethodInfo> accessors = new();
    private readonly AdapterMethodTranslator adapters = new(helpers);

    internal bool TryEmit(ILGenerator il, MethodInfo method)
    {
        if (adapters.TryEmit(il, method))
            return true;
        var type = method.DeclaringType!;
        if (type.IsArray && CapturedCollection.ElementType(type) is not null)
        {
            if (method.Name is not ("Get" or "Set" or "Address"))
                throw new NotSupportedException($"Array operation '{method.Name}' is unsupported inside GPU loops.");
            il.Emit(OpCodes.Call, Accessor(method));
            return true;
        }
        if (type == typeof(Array))
        {
            string name = method.Name switch
            {
                "get_Length" => nameof(DeviceCollection.Count),
                "get_LongLength" => nameof(DeviceCollection.LongCount),
                "get_Rank" => nameof(DeviceCollection.GetRank),
                "GetLength" => nameof(DeviceCollection.GetLength),
                "GetLongLength" => nameof(DeviceCollection.GetLongLength),
                "GetLowerBound" => nameof(DeviceCollection.GetLowerBound),
                "GetUpperBound" => nameof(DeviceCollection.GetUpperBound),
                _ => throw new NotSupportedException($"Array operation '{method.Name}' is unsupported inside GPU loops.")
            };
            il.Emit(OpCodes.Call, typeof(DeviceCollection).GetMethod(name)!);
            return true;
        }
        if (type.IsGenericType)
        {
            var definition = type.GetGenericTypeDefinition();
            if (definition == typeof(List<>) || definition == typeof(IList<>) ||
                definition == typeof(IReadOnlyList<>) || definition == typeof(ICollection<>) ||
                definition == typeof(IReadOnlyCollection<>))
            {
                string name = method.Name switch
                {
                    "get_Item" => nameof(DeviceCollection.Read),
                    "set_Item" => nameof(DeviceCollection.Write),
                    "get_Count" => nameof(DeviceCollection.Count),
                    _ => throw new NotSupportedException($"Collection operation '{method.Name}' requires an explicit GPU operation.")
                };
                var replacement = typeof(DeviceCollection).GetMethod(name)!;
                if (replacement.IsGenericMethodDefinition)
                    replacement = replacement.MakeGenericMethod(DeviceValueType.MapType(type.GetGenericArguments()[0]));
                il.Emit(OpCodes.Call, replacement);
                return true;
            }
        }
        return false;
    }

    internal bool TryEmitConstructor(ILGenerator il, OpCode code, ConstructorInfo constructor) =>
        adapters.TryEmitConstructor(il, code, constructor);

    private MethodInfo Accessor(MethodInfo original)
    {
        if (accessors.TryGetValue(original, out var existing))
            return existing;
        var type = original.DeclaringType!;
        int rank = type.GetArrayRank();
        var element = type.GetElementType()!;
        bool node = element.IsArray;
        bool write = original.Name == "Set";
        bool address = original.Name == "Address";
        if (node && (write || address))
            throw new NotSupportedException("Changing array topology requires an explicit GPU object pool.");
        var mapped = node ? typeof(DeviceCollection) : DeviceValueType.MapType(element);
        var parameters = new List<Type> { typeof(DeviceCollection) };
        parameters.AddRange(Enumerable.Repeat(typeof(int), rank));
        if (write)
            parameters.Add(mapped);
        var result = write ? typeof(void) : address ? mapped.MakeByRefType() : mapped;
        var builder = helpers.DefineMethod("Array" + accessors.Count,
            MethodAttributes.Public | MethodAttributes.Static, result, parameters.ToArray());
        accessors.Add(original, builder);
        var il = builder.GetILGenerator();
        var offset = il.DeclareLocal(typeof(int));
        il.Emit(OpCodes.Ldc_I4_0);
        il.Emit(OpCodes.Stloc, offset);
        for (int dimension = 0; dimension < rank; dimension++)
        {
            il.Emit(OpCodes.Ldloc, offset);
            il.Emit(OpCodes.Ldarg_0);
            il.Emit(OpCodes.Ldc_I4, dimension);
            il.Emit(OpCodes.Ldarg, (short)(dimension + 1));
            il.Emit(OpCodes.Call, typeof(DeviceCollection).GetMethod(nameof(DeviceCollection.DimensionOffset))!);
            il.Emit(OpCodes.Add);
            il.Emit(OpCodes.Stloc, offset);
        }
        il.Emit(OpCodes.Ldarg_0);
        il.Emit(OpCodes.Ldloc, offset);
        if (write)
            il.Emit(OpCodes.Ldarg, (short)(rank + 1));
        var operation = typeof(DeviceCollection).GetMethod(node ? nameof(DeviceCollection.Row) :
            write ? nameof(DeviceCollection.Write) : address ? nameof(DeviceCollection.Address) : nameof(DeviceCollection.Read))!;
        il.Emit(OpCodes.Call, node ? operation : operation.MakeGenericMethod(mapped));
        il.Emit(OpCodes.Ret);
        return builder;
    }
}

