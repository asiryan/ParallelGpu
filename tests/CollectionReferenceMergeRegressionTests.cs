using System.Reflection;
using System.Reflection.Emit;
using Xunit;

namespace ParallelGpu.Tests;

[Collection("GPU")]
public sealed class CollectionReferenceMergeRegressionTests(GpuFixture fixture)
{
    [Fact]
    public void CachedNullCoalescingCapturesPreserveNullTestsIdentityAndInterfaceIndexing()
    {
        IReadOnlyList<int>? list = null;
        int[]? array = null;
        int[] nullPattern = new int[2], nonNullPattern = new int[2];
        int[] equalNull = new int[2], unequalNull = new int[2];
        int[] arrayIdentity = new int[2], listIdentity = new int[2], values = new int[2];
        Action<int> body = i =>
        {
            nullPattern[i] = (list ?? array) is null ? 1 : 0;
            nonNullPattern[i] = (list ?? array) is not null ? 1 : 0;
            equalNull[i] = (list ?? array) == null ? 1 : 0;
            unequalNull[i] = (list ?? array) != null ? 1 : 0;
            arrayIdentity[i] = (list ?? array) == array ? 1 : 0;
            listIdentity[i] = (list ?? array) == list ? 1 : 0;
            values[i] = (list ?? array) is null ? -1 : (list ?? array)![i];
        };

        Verify([1, 1], [1, 1], [1, 1], [-1, -1]);
        array = [7, 11];
        Verify([0, 0], [1, 1], [0, 0], [7, 11]);
        list = new List<int> { 13, 17 };
        Verify([0, 0], [0, 0], [1, 1], [13, 17]);
        list = array;
        Verify([0, 0], [1, 1], [1, 1], [7, 11]);
        list = null;
        array = null;
        Verify([1, 1], [1, 1], [1, 1], [-1, -1]);

        // A correct translation must also leave the same executor usable by later bodies.
        fixture.Gpu.For(0, values.Length, i => values[i] = i + 23);
        Assert.Equal(new[] { 23, 24 }, values);

        void Verify(int[] expectedNulls, int[] expectedArrayIdentity, int[] expectedListIdentity, int[] expectedValues)
        {
            fixture.Gpu.For(0, values.Length, body);
            Assert.Equal(expectedNulls, nullPattern);
            Assert.Equal(expectedNulls.Select(value => 1 - value), nonNullPattern);
            Assert.Equal(expectedNulls, equalNull);
            Assert.Equal(expectedNulls.Select(value => 1 - value), unequalNull);
            Assert.Equal(expectedArrayIdentity, arrayIdentity);
            Assert.Equal(expectedListIdentity, listIdentity);
            Assert.Equal(expectedValues, values);
        }
    }

    [Fact]
    public void ChainedCoalescingAcrossListInterfaceAndArrayPreservesFirstNonNullReference()
    {
        List<int>? concrete = null;
        IReadOnlyList<int>? middle = null;
        int[]? array = null;
        int[] nulls = new int[2], identities = new int[2], values = new int[2];
        Action<int> body = i =>
        {
            nulls[i] = (concrete ?? middle ?? array) is null ? 1 : 0;
            identities[i] = (concrete ?? middle ?? array) == middle ? 1 : 0;
            values[i] = (concrete ?? middle ?? array) is null ? -1 : (concrete ?? middle ?? array)![i];
        };

        Verify([1, 1], [1, 1], [-1, -1]);
        array = [2, 3];
        Verify([0, 0], [0, 0], [2, 3]);
        middle = new List<int> { 5, 7 };
        Verify([0, 0], [1, 1], [5, 7]);
        concrete = new List<int> { 11, 13 };
        Verify([0, 0], [0, 0], [11, 13]);
        middle = concrete;
        Verify([0, 0], [1, 1], [11, 13]);

        void Verify(int[] expectedNulls, int[] expectedIdentities, int[] expectedValues)
        {
            fixture.Gpu.For(0, values.Length, body);
            Assert.Equal(expectedNulls, nulls);
            Assert.Equal(expectedIdentities, identities);
            Assert.Equal(expectedValues, values);
        }
    }

    [Fact]
    public void TranslatedHelpersPreserveCoalescedNullTestsIdentityAndReturnedCollections()
    {
        IReadOnlyList<int>? list = null;
        int[]? array = null;
        int[] nulls = new int[2], identities = new int[2], values = new int[2];
        Action<int> body = i =>
        {
            nulls[i] = IsMissing(list, array) ? 1 : 0;
            identities[i] = HasIdentity(list, array, array) ? 1 : 0;
            values[i] = IsMissing(list, array) ? -1 : Choose(list, array)![i];
        };

        Verify([1, 1], [1, 1], [-1, -1]);
        array = [17, 19];
        Verify([0, 0], [1, 1], [17, 19]);
        list = new List<int> { 23, 29 };
        Verify([0, 0], [0, 0], [23, 29]);
        list = array;
        Verify([0, 0], [1, 1], [17, 19]);

        void Verify(int[] expectedNulls, int[] expectedIdentities, int[] expectedValues)
        {
            fixture.Gpu.For(0, values.Length, body);
            Assert.Equal(expectedNulls, nulls);
            Assert.Equal(expectedIdentities, identities);
            Assert.Equal(expectedValues, values);
        }
    }

    [Theory]
    [InlineData(false, false)]
    [InlineData(true, false)]
    [InlineData(false, true)]
    [InlineData(true, true)]
    public void CollectionStackMergeSupportsComparisonAndBranchNullTests(bool useBranch, bool nestedArrays)
    {
        // Keep unlike original stack types at the merge; Roslyn can insert a common-type
        // local for conditional expressions and conceal this branch form from the translator.
        TypeBuilder owner = AssemblyBuilder.DefineDynamicAssembly(
            new AssemblyName("CollectionReferenceMerge" + Guid.NewGuid().ToString("N")), AssemblyBuilderAccess.Run)
            .DefineDynamicModule("Regression").DefineType("Loop", TypeAttributes.Public);
        FieldBuilder first = owner.DefineField("First", nestedArrays ? typeof(int[][]) : typeof(int[]),
            FieldAttributes.Public | FieldAttributes.Static);
        FieldBuilder second = owner.DefineField("Second", nestedArrays ? typeof(float[][]) : typeof(List<int>),
            FieldAttributes.Public | FieldAttributes.Static);
        FieldBuilder output = owner.DefineField("Output", typeof(int[]), FieldAttributes.Public | FieldAttributes.Static);
        var method = owner.DefineMethod("Run", MethodAttributes.Public | MethodAttributes.Static,
            typeof(void), [typeof(int)]);
        var il = method.GetILGenerator();
        var chooseFirst = il.DefineLabel();
        var merge = il.DefineLabel();
        il.Emit(OpCodes.Ldsfld, output);
        il.Emit(OpCodes.Ldarg_0);
        il.Emit(OpCodes.Ldarg_0);
        il.Emit(OpCodes.Brfalse_S, chooseFirst);
        il.Emit(OpCodes.Ldsfld, second);
        il.Emit(OpCodes.Br_S, merge);
        il.MarkLabel(chooseFirst);
        il.Emit(OpCodes.Ldsfld, first);
        il.MarkLabel(merge);
        if (nestedArrays)
        {
            // The joined containers have different ultimate element types. Their selected
            // leaves must still retain the collection category after ldelem.ref.
            il.Emit(OpCodes.Ldc_I4_0);
            il.Emit(OpCodes.Ldelem_Ref);
        }
        if (useBranch)
        {
            var isNull = il.DefineLabel();
            var store = il.DefineLabel();
            il.Emit(OpCodes.Brfalse_S, isNull);
            il.Emit(OpCodes.Ldc_I4_0);
            il.Emit(OpCodes.Br_S, store);
            il.MarkLabel(isNull);
            il.Emit(OpCodes.Ldc_I4_1);
            il.MarkLabel(store);
        }
        else
        {
            il.Emit(OpCodes.Ldnull);
            il.Emit(OpCodes.Ceq);
        }
        il.Emit(OpCodes.Stelem_I4);
        il.Emit(OpCodes.Ret);
        Type created = owner.CreateType()!;
        int[] result = new int[2];
        created.GetField("First")!.SetValue(null, nestedArrays ? new int[][] { new[] { 31 } } : new[] { 31 });
        created.GetField("Second")!.SetValue(null, nestedArrays ? new float[][] { null! } : null);
        created.GetField("Output")!.SetValue(null, result);
        var body = (Action<int>)created.GetMethod("Run")!.CreateDelegate(typeof(Action<int>));
        body(0);
        body(1);
        Assert.Equal(new[] { 0, 1 }, result);
        Array.Fill(result, 41);
        fixture.Gpu.For(0, result.Length, body);
        Assert.Equal(new[] { 0, 1 }, result);
        created.GetField("First")!.SetValue(null, nestedArrays ? new int[][] { null! } : null);
        created.GetField("Second")!.SetValue(null,
            nestedArrays ? new float[][] { new[] { 37f } } : new List<int> { 37 });
        fixture.Gpu.For(0, result.Length, body);
        Assert.Equal(new[] { 1, 0 }, result);
    }

    private static bool IsMissing(IReadOnlyList<int>? list, int[]? array) => (list ?? array) is null;
    private static bool HasIdentity(IReadOnlyList<int>? list, int[]? array, IReadOnlyList<int>? candidate) =>
        (list ?? array) == candidate;
    private static IReadOnlyList<int>? Choose(IReadOnlyList<int>? list, int[]? array) => list ?? array;
}
