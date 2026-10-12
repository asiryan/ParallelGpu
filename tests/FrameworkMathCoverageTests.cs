using System.Reflection;
using System.Reflection.Emit;
using Xunit;

namespace ParallelGpu.Tests;

[Collection("GPU")]
public sealed class FrameworkMathCoverageTests(GpuFixture fixture)
{
    private static readonly Type[] NumericTypes =
        [typeof(sbyte), typeof(byte), typeof(short), typeof(ushort), typeof(int), typeof(uint),
            typeof(long), typeof(ulong), typeof(float), typeof(double)];

    [Fact]
    public void EveryFrameworkMathSignatureForSupportedNumbersExecutesOnDevice()
    {
        MethodInfo[] methods = new[] { typeof(Math), typeof(MathF) }
            .SelectMany(owner => owner.GetMethods(BindingFlags.Public | BindingFlags.Static))
            .Where(method => SupportedResult(method.ReturnType) && method.GetParameters()
                .All(parameter => SupportedParameter(parameter.ParameterType)))
            .OrderBy(method => method.DeclaringType!.Name).ThenBy(method => method.Name).ToArray();
        Assert.NotEmpty(methods);
        var assembly = AssemblyBuilder.DefineDynamicAssembly(new AssemblyName("MathCoverage" + Guid.NewGuid().ToString("N")),
            AssemblyBuilderAccess.Run);
        var module = assembly.DefineDynamicModule("Loops");
        for (int index = 0; index < methods.Length; index++)
            ExecuteSignature(module, methods[index], index);
    }

    private void ExecuteSignature(ModuleBuilder module, MethodInfo method, int sequence)
    {
        var type = module.DefineType("Loop" + sequence, TypeAttributes.Public | TypeAttributes.Abstract | TypeAttributes.Sealed);
        var outputField = type.DefineField("Output", method.ReturnType.MakeArrayType(), FieldAttributes.Public | FieldAttributes.Static);
        var body = type.DefineMethod("Body", MethodAttributes.Public | MethodAttributes.Static,
            typeof(void), [typeof(int)]);
        var il = body.GetILGenerator();
        var parameters = method.GetParameters();
        var arguments = new object?[parameters.Length];
        il.Emit(OpCodes.Ldsfld, outputField);
        il.Emit(OpCodes.Ldarg_0);
        for (int index = 0; index < parameters.Length; index++)
        {
            Type parameter = parameters[index].ParameterType;
            if (parameter.IsByRef)
            {
                Type element = parameter.GetElementType()!;
                var local = il.DeclareLocal(element);
                il.Emit(OpCodes.Ldloca, local);
                il.Emit(OpCodes.Initobj, element);
                il.Emit(OpCodes.Ldloca, local);
                arguments[index] = Activator.CreateInstance(element);
            }
            else
            {
                var field = type.DefineField("Argument" + index, parameter, FieldAttributes.Public | FieldAttributes.Static);
                il.Emit(OpCodes.Ldsfld, field);
                arguments[index] = parameter == typeof(MidpointRounding) ? MidpointRounding.ToEven :
                    Convert.ChangeType(index == 0 ? 1.25 : index + 1, parameter);
            }
        }
        il.Emit(OpCodes.Call, method);
        il.Emit(OpCodes.Stelem, method.ReturnType);
        il.Emit(OpCodes.Ret);
        Type loop = type.CreateType()!;
        Array output = Array.CreateInstance(method.ReturnType, 1);
        loop.GetField("Output")!.SetValue(null, output);
        for (int index = 0; index < parameters.Length; index++)
            if (!parameters[index].ParameterType.IsByRef)
                loop.GetField("Argument" + index)!.SetValue(null, arguments[index]);
        var callback = (Action<int>)loop.GetMethod("Body")!.CreateDelegate(typeof(Action<int>));
        fixture.Gpu.For(0, 1, callback);
        object expected = method.Invoke(null, arguments)!;
        AssertResult(expected, output.GetValue(0)!, method.ToString()!);
    }

    private static bool SupportedParameter(Type type) => type.IsByRef
        ? NumericTypes.Contains(type.GetElementType()!) : NumericTypes.Contains(type) || type == typeof(MidpointRounding);

    private static bool SupportedResult(Type type) => NumericTypes.Contains(type) ||
        type.IsGenericType && type.GetGenericTypeDefinition() == typeof(ValueTuple<,>) &&
        type.GetGenericArguments().All(NumericTypes.Contains);

    private static void AssertResult(object expected, object actual, string context)
    {
        // Estimate APIs may use a roughly 12-bit hardware approximation on the host;
        // a more accurate device result still satisfies their numerical contract.
        bool estimate = context.Contains("Estimate", StringComparison.Ordinal);
        if (expected is double value64)
            AssertFloating(value64, (double)actual, estimate ? 5e-4 : 4e-12, context);
        else if (expected is float value32)
            AssertFloating(value32, (float)actual, estimate ? 5e-4 : 4e-6, context);
        else if (expected.GetType().IsGenericType)
            foreach (var field in expected.GetType().GetFields(BindingFlags.Instance | BindingFlags.Public))
                AssertResult(field.GetValue(expected)!, field.GetValue(actual)!, context + "." + field.Name);
        else
            Assert.True(expected.Equals(actual), $"{context}: expected {expected}, actual {actual}");
    }

    private static void AssertFloating(double expected, double actual, double relativeTolerance, string context)
    {
        if (double.IsNaN(expected))
            Assert.True(double.IsNaN(actual), context);
        else if (double.IsInfinity(expected) || expected == 0)
            Assert.Equal(BitConverter.DoubleToInt64Bits(expected), BitConverter.DoubleToInt64Bits(actual));
        else
            Assert.True(double.IsFinite(actual) && Math.Abs(actual - expected) <= Math.Abs(expected) * relativeTolerance,
                $"{context}: expected {expected:R}, actual {actual:R}");
    }
}
