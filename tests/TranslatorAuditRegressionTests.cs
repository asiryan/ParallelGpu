using System.Reflection;
using System.Reflection.Emit;
using Xunit;

namespace ParallelGpu.Tests;

[Collection("GPU")]
public sealed class TranslatorAuditRegressionTests(GpuFixture fixture)
{
    [Fact]
    public void NullableBooleanValueSeparatesPresenceFromPayloadAndRecoversAfterNull()
    {
        bool?[] input = [false, true];
        bool[] output = [true, false];
        int[] writes = [41, 43];
        Action<int> body = index =>
        {
            writes[index] = 99;
            output[index] = input[index]!.Value;
        };
        fixture.Gpu.For(0, input.Length, body);
        Assert.Equal(new[] { false, true }, output);
        Assert.Equal(new[] { 99, 99 }, writes);

        input[1] = null;
        writes[0] = 41;
        writes[1] = 43;
        Assert.Throws<InvalidOperationException>(() => fixture.Gpu.For(0, input.Length, body));
        Assert.Equal(new[] { false, true }, output);
        Assert.Equal(new[] { 41, 43 }, writes);

        input[0] = true;
        input[1] = false;
        fixture.Gpu.For(0, input.Length, body);
        Assert.Equal(new[] { true, false }, output);
        fixture.Gpu.For(0, input.Length, index => output[index] = ReadNullableBoolean(input[index]));
        Assert.Equal(new[] { true, false }, output);
    }

    [Fact]
    public void TransitiveHelperCanReadPrivateMappedStaticValueFromAnotherAssembly()
    {
        // The loop references A, while A calls B. B's readonly char needs a mapped static initializer.
        TypeBuilder ownerB = CreateModule("PrivateConstant").DefineType("Constants", TypeAttributes.Public);
        FieldBuilder character = ownerB.DefineField("Character", typeof(char),
            FieldAttributes.Private | FieldAttributes.Static | FieldAttributes.InitOnly);
        var initializer = ownerB.DefineTypeInitializer().GetILGenerator();
        initializer.Emit(OpCodes.Ldc_I4, (int)'X');
        initializer.Emit(OpCodes.Stsfld, character);
        initializer.Emit(OpCodes.Ret);
        var readB = ownerB.DefineMethod("Read", MethodAttributes.Public | MethodAttributes.Static,
            typeof(char), Type.EmptyTypes);
        var readBIl = readB.GetILGenerator();
        readBIl.Emit(OpCodes.Ldsfld, character);
        readBIl.Emit(OpCodes.Ret);
        Type typeB = ownerB.CreateType()!;

        TypeBuilder ownerA = CreateModule("TransitiveHelper").DefineType("Helpers", TypeAttributes.Public);
        var readA = ownerA.DefineMethod("Read", MethodAttributes.Public | MethodAttributes.Static,
            typeof(int), Type.EmptyTypes);
        var readAIl = readA.GetILGenerator();
        readAIl.Emit(OpCodes.Call, typeB.GetMethod("Read")!);
        readAIl.Emit(OpCodes.Ret);
        Type typeA = ownerA.CreateType()!;
        int expected = (int)typeA.GetMethod("Read")!.Invoke(null, null)!;

        TypeBuilder loopType = CreateModule("LoopOwner").DefineType("Loops", TypeAttributes.Public);
        FieldBuilder resultField = loopType.DefineField("Result", typeof(int[]),
            FieldAttributes.Public | FieldAttributes.Static);
        var loop = loopType.DefineMethod("Run", MethodAttributes.Public | MethodAttributes.Static,
            typeof(void), [typeof(int)]);
        var loopIl = loop.GetILGenerator();
        loopIl.Emit(OpCodes.Ldsfld, resultField);
        loopIl.Emit(OpCodes.Ldarg_0);
        loopIl.Emit(OpCodes.Call, typeA.GetMethod("Read")!);
        loopIl.Emit(OpCodes.Stelem_I4);
        loopIl.Emit(OpCodes.Ret);
        Type createdLoop = loopType.CreateType()!;
        FieldInfo destination = createdLoop.GetField("Result")!;
        var body = (Action<int>)createdLoop.GetMethod("Run")!.CreateDelegate(typeof(Action<int>));

        int[] first = [41];
        destination.SetValue(null, first);
        fixture.Gpu.For(0, first.Length, body);
        Assert.Equal(expected, first[0]);

        int[] second = [43, 47, 53];
        destination.SetValue(null, second);
        fixture.Gpu.For(0, second.Length, body);
        Assert.Equal(Enumerable.Repeat(expected, second.Length), second);
        Assert.Equal(expected, first[0]);
    }

    private static ModuleBuilder CreateModule(string prefix) => AssemblyBuilder.DefineDynamicAssembly(
        new AssemblyName(prefix + Guid.NewGuid().ToString("N")), AssemblyBuilderAccess.Run).DefineDynamicModule(prefix);

    private static bool ReadNullableBoolean(bool? value) => value!.Value;
}
