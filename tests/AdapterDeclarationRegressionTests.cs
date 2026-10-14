using Xunit;

namespace ParallelGpu.Tests;

[Collection("GPU")]
public sealed class AdapterDeclarationRegressionTests(GpuFixture fixture)
{
    private Gpu Gpu => fixture.Gpu;

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void ExplicitInterfaceIndexersKeepTheirDeclaredStorage(bool derivedFirst)
    {
        InterfaceOwner.Register();
        var owner = new InterfaceOwner(7, 100);
        IBaseView baseView = owner;
        IDerivedView derivedView = owner;
        int[] observed = [-1, -1, -1, -1];
        if (derivedFirst)
            Gpu.For<int>(0, 1, i =>
            {
                observed[1] = derivedView[i];
                observed[0] = baseView[i];
                derivedView[i] += 20;
                baseView[i] += 10;
                observed[3] = derivedView[i];
                observed[2] = baseView[i];
            });
        else
            Gpu.For<int>(0, 1, i =>
            {
                observed[0] = baseView[i];
                observed[1] = derivedView[i];
                baseView[i] += 10;
                derivedView[i] += 20;
                observed[2] = baseView[i];
                observed[3] = derivedView[i];
            });
        Assert.Equal(new[] { 7, 100, 17, 120 }, observed);
        Assert.Equal(new[] { 17 }, owner.BaseStorage);
        Assert.Equal(new[] { 120 }, owner.DerivedStorage);
        Assert.Equal(0, owner.HostIndexerCalls);
    }

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void InterfaceHelpersAndCachedInvocationKeepFreshDeclaredStorage(bool derivedFirst)
    {
        InterfaceOwner.Register();
        var first = new InterfaceOwner(7, 100);
        IBaseView baseView = first;
        IDerivedView derivedView = first;
        int[] observed = [-1, -1, -1, -1];
        Action<int> body;
        if (derivedFirst)
            body = i =>
            {
                observed[1] = ReadDerived(derivedView, i);
                observed[0] = ReadBase(baseView, i);
                WriteDerived(derivedView, i, ReadDerived(derivedView, i) + 20);
                WriteBase(baseView, i, ReadBase(baseView, i) + 10);
                observed[3] = ReadDerived(derivedView, i);
                observed[2] = ReadBase(baseView, i);
            };
        else
            body = i =>
            {
                observed[0] = ReadBase(baseView, i);
                observed[1] = ReadDerived(derivedView, i);
                WriteBase(baseView, i, ReadBase(baseView, i) + 10);
                WriteDerived(derivedView, i, ReadDerived(derivedView, i) + 20);
                observed[2] = ReadBase(baseView, i);
                observed[3] = ReadDerived(derivedView, i);
            };
        Gpu.For<int>(0, 1, body);
        Assert.Equal(new[] { 7, 100, 17, 120 }, observed);

        var second = new InterfaceOwner(70, 1000);
        baseView = second;
        derivedView = second;
        Array.Fill(observed, -1);
        Gpu.For<int>(0, 1, body);
        Assert.Equal(new[] { 70, 1000, 80, 1020 }, observed);
        Assert.Equal(new[] { 17 }, first.BaseStorage);
        Assert.Equal(new[] { 120 }, first.DerivedStorage);
        Assert.Equal(new[] { 80 }, second.BaseStorage);
        Assert.Equal(new[] { 1020 }, second.DerivedStorage);
        Assert.Equal(0, first.HostIndexerCalls);
        Assert.Equal(0, second.HostIndexerCalls);
    }

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void DeclaredReadOnlyInterfaceRejectsWritesAlongsideWritableDerivedInterface(bool derivedFirst)
    {
        RestrictedOwner.Register();
        var owner = new RestrictedOwner();
        IRestrictedBase baseView = owner;
        IRestrictedDerived derivedView = owner;
        int[] observed = [-1, -1];
        if (derivedFirst)
            Assert.Throws<InvalidOperationException>(() => Gpu.For<int>(0, 1, i =>
            {
                derivedView[i] += 20;
                observed[1] = derivedView[i];
                observed[0] = baseView[i];
                baseView[i] = 9;
            }));
        else
            Assert.Throws<InvalidOperationException>(() => Gpu.For<int>(0, 1, i =>
            {
                observed[0] = baseView[i];
                derivedView[i] += 20;
                observed[1] = derivedView[i];
                baseView[i] = 9;
            }));
        Assert.Equal(new[] { 7 }, owner.BaseStorage);
        Assert.Equal(new[] { 100 }, owner.DerivedStorage);
        Assert.Equal(new[] { -1, -1 }, observed);

        if (derivedFirst)
            Gpu.For<int>(0, 1, i =>
            {
                derivedView[i] += 20;
                observed[1] = derivedView[i];
                observed[0] = baseView[i];
            });
        else
            Gpu.For<int>(0, 1, i =>
            {
                observed[0] = baseView[i];
                derivedView[i] += 20;
                observed[1] = derivedView[i];
            });
        Assert.Equal(new[] { 7 }, owner.BaseStorage);
        Assert.Equal(new[] { 120 }, owner.DerivedStorage);
        Assert.Equal(new[] { 7, 120 }, observed);
        Assert.Equal(0, owner.HostIndexerCalls);
    }

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void HiddenClassIndexersKeepTheirDeclaredStorage(bool derivedFirst)
    {
        DerivedClassView.Register();
        var derivedView = new DerivedClassView();
        BaseClassView baseView = derivedView;
        int[] observed = [-1, -1, -1, -1];
        if (derivedFirst)
            Gpu.For<int>(0, 1, i =>
            {
                observed[1] = derivedView[i];
                observed[0] = baseView[i];
                derivedView[i] += 20;
                baseView[i] += 10;
                observed[3] = derivedView[i];
                observed[2] = baseView[i];
            });
        else
            Gpu.For<int>(0, 1, i =>
            {
                observed[0] = baseView[i];
                observed[1] = derivedView[i];
                baseView[i] += 10;
                derivedView[i] += 20;
                observed[2] = baseView[i];
                observed[3] = derivedView[i];
            });
        Assert.Equal(new[] { 7, 100, 17, 120 }, observed);
        Assert.Equal(new[] { 17 }, baseView.BaseStorage);
        Assert.Equal(new[] { 120 }, derivedView.DerivedStorage);
        Assert.Equal(0, baseView.HostIndexerCalls);
    }

    [Fact]
    public void HiddenInterfaceIndexerConversionThroughHelperArgumentIsRejected()
    {
        InterfaceOwner.Register();
        var owner = new InterfaceOwner(7, 100);
        IDerivedView derivedView = owner;
        int[] marker = [-1];
        var error = Assert.Throws<NotSupportedException>(() => Gpu.For<int>(0, 1, i =>
        {
            marker[i] = 99;
            derivedView[i] += 20;
            marker[i] = ReadBase(derivedView, i);
        }));
        Assert.Contains("indexer", error.Message);
        AssertRejectedConversionLeftExecutorUsable(owner, marker);
    }

    [Fact]
    public void HiddenInterfaceIndexerConversionThroughLocalIsRejected()
    {
        InterfaceOwner.Register();
        var owner = new InterfaceOwner(7, 100);
        IDerivedView derivedView = owner;
        int[] marker = [-1];
        var error = Assert.Throws<NotSupportedException>(() => Gpu.For<int>(0, 1, i =>
        {
            marker[i] = 99;
            derivedView[i] += 20;
            IBaseView baseView = derivedView;
            marker[i] = baseView[i];
        }));
        Assert.Contains("indexer", error.Message);
        AssertRejectedConversionLeftExecutorUsable(owner, marker);
    }

    [Fact]
    public void HiddenInterfaceIndexerConversionThroughHelperReturnIsRejected()
    {
        InterfaceOwner.Register();
        var owner = new InterfaceOwner(7, 100);
        IDerivedView derivedView = owner;
        int[] marker = [-1];
        var error = Assert.Throws<NotSupportedException>(() => Gpu.For<int>(0, 1, i =>
        {
            marker[i] = 99;
            derivedView[i] += 20;
            marker[i] = ReturnBase(derivedView)[i];
        }));
        Assert.Contains("indexer", error.Message);
        AssertRejectedConversionLeftExecutorUsable(owner, marker);
    }

    [Fact]
    public void HiddenInterfaceIndexerConversionAtControlFlowJoinIsRejected()
    {
        InterfaceOwner.Register();
        var owner = new InterfaceOwner(7, 100);
        IBaseView baseView = owner;
        IDerivedView derivedView = owner;
        int[] marker = [-1, -1];
        var error = Assert.Throws<NotSupportedException>(() => Gpu.For<int>(0, 2, i =>
        {
            marker[i] = 99;
            IBaseView selected = i == 0 ? baseView : derivedView;
            marker[i] = selected[0];
        }));
        Assert.Contains("indexer", error.Message);
        AssertRejectedConversionLeftExecutorUsable(owner, marker);
    }

    [Fact]
    public void HiddenInterfaceIndexerConversionThroughOptimizedUpcastIsRejected()
    {
        InterfaceOwner.Register();
        var owner = new InterfaceOwner(7, 100);
        IDerivedView derivedView = owner;
        int[] marker = [-1];
        var error = Assert.Throws<NotSupportedException>(() => Gpu.For<int>(0, 1, i =>
        {
            marker[i] = 99;
            derivedView[i] += 20;
            marker[i] = ((IBaseView)derivedView)[i];
        }));
        Assert.Contains("indexer", error.Message);
        AssertRejectedConversionLeftExecutorUsable(owner, marker);
    }

    [Fact]
    public void ConcreteAdapterCanUseInterfaceHelpersForTheSameImplementedIndexer()
    {
        SharedClassView.Register();
        var owner = new SharedClassView();
        int[] observed = [-1, -1, -1];
        Gpu.For<int>(0, 1, i =>
        {
            observed[0] = ReadShared(owner, i);
            owner[i] += 10;
            observed[1] = ReadShared(owner, i);
            WriteShared(owner, i, ReadShared(owner, i) + 20);
            observed[2] = owner[i];
        });
        Assert.Equal(new[] { 7, 17, 37 }, observed);
        Assert.Equal(new[] { 37 }, owner.Storage);
        Assert.Equal(0, owner.HostIndexerCalls);
    }

    private void AssertRejectedConversionLeftExecutorUsable(InterfaceOwner owner, int[] marker)
    {
        Assert.All(marker, value => Assert.Equal(-1, value));
        Assert.Equal(new[] { 7 }, owner.BaseStorage);
        Assert.Equal(new[] { 100 }, owner.DerivedStorage);
        Assert.Equal(0, owner.HostIndexerCalls);
        IBaseView baseView = owner;
        IDerivedView derivedView = owner;
        int[] observed = [-1, -1];
        Gpu.For<int>(0, 1, i =>
        {
            observed[0] = baseView[i];
            observed[1] = derivedView[i];
        });
        Assert.Equal(new[] { 7, 100 }, observed);
    }

    private static int ReadBase(IBaseView view, int index) => view[index];
    private static int ReadDerived(IDerivedView view, int index) => view[index];
    private static void WriteBase(IBaseView view, int index, int value) => view[index] = value;
    private static void WriteDerived(IDerivedView view, int index, int value) => view[index] = value;
    private static IBaseView ReturnBase(IDerivedView view) => view;
    private static int ReadShared(ISharedView view, int index) => view[index];
    private static void WriteShared(ISharedView view, int index, int value) => view[index] = value;

    private interface ISharedView
    {
        int[] Storage { get; }
        int this[int index] { get; set; }
    }

    private sealed class SharedClassView : ISharedView
    {
        private static readonly Lazy<bool> registered = new(() =>
        {
            GpuCollectionAdapters.Register<ISharedView, int>(view => new(view.Storage));
            GpuCollectionAdapters.Register<SharedClassView, int>(view => new(view.Storage));
            return true;
        });

        public int[] Storage { get; } = [7];
        public int HostIndexerCalls;
        public int this[int index]
        {
            get { HostIndexerCalls++; return Storage[index]; }
            set { HostIndexerCalls++; Storage[index] = value; }
        }
        public static void Register() => _ = registered.Value;
    }

    private interface IBaseView
    {
        int[] BaseStorage { get; }
        int this[int index] { get; set; }
    }

    private interface IDerivedView : IBaseView
    {
        int[] DerivedStorage { get; }
        new int this[int index] { get; set; }
    }

    private sealed class InterfaceOwner(int baseValue, int derivedValue) : IDerivedView
    {
        private static readonly Lazy<bool> registered = new(() =>
        {
            GpuCollectionAdapters.Register<IBaseView, int>(view => new(view.BaseStorage));
            GpuCollectionAdapters.Register<IDerivedView, int>(view => new(view.DerivedStorage));
            return true;
        });

        public int[] BaseStorage { get; } = [baseValue];
        public int[] DerivedStorage { get; } = [derivedValue];
        public int HostIndexerCalls;
        int IBaseView.this[int index]
        {
            get { HostIndexerCalls++; return BaseStorage[index]; }
            set { HostIndexerCalls++; BaseStorage[index] = value; }
        }
        int IDerivedView.this[int index]
        {
            get { HostIndexerCalls++; return DerivedStorage[index]; }
            set { HostIndexerCalls++; DerivedStorage[index] = value; }
        }
        public static void Register() => _ = registered.Value;
    }

    private interface IRestrictedBase
    {
        int[] BaseStorage { get; }
        int this[int index] { get; set; }
    }

    private interface IRestrictedDerived : IRestrictedBase
    {
        int[] DerivedStorage { get; }
        new int this[int index] { get; set; }
    }

    private sealed class RestrictedOwner : IRestrictedDerived
    {
        private static readonly Lazy<bool> registered = new(() =>
        {
            GpuCollectionAdapters.Register<IRestrictedBase, int>(view => new(view.BaseStorage, true));
            GpuCollectionAdapters.Register<IRestrictedDerived, int>(view => new(view.DerivedStorage));
            return true;
        });

        public int[] BaseStorage { get; } = [7];
        public int[] DerivedStorage { get; } = [100];
        public int HostIndexerCalls;
        int IRestrictedBase.this[int index]
        {
            get { HostIndexerCalls++; return BaseStorage[index]; }
            set { HostIndexerCalls++; BaseStorage[index] = value; }
        }
        int IRestrictedDerived.this[int index]
        {
            get { HostIndexerCalls++; return DerivedStorage[index]; }
            set { HostIndexerCalls++; DerivedStorage[index] = value; }
        }
        public static void Register() => _ = registered.Value;
    }

    private class BaseClassView
    {
        public int[] BaseStorage { get; } = [7];
        public int HostIndexerCalls;
        public int this[int index]
        {
            get { HostIndexerCalls++; return BaseStorage[index]; }
            set { HostIndexerCalls++; BaseStorage[index] = value; }
        }
    }

    private sealed class DerivedClassView : BaseClassView
    {
        private static readonly Lazy<bool> registered = new(() =>
        {
            GpuCollectionAdapters.Register<BaseClassView, int>(view => new(view.BaseStorage));
            GpuCollectionAdapters.Register<DerivedClassView, int>(view => new(view.DerivedStorage));
            return true;
        });

        public int[] DerivedStorage { get; } = [100];
        public new int this[int index]
        {
            get { HostIndexerCalls++; return DerivedStorage[index]; }
            set { HostIndexerCalls++; DerivedStorage[index] = value; }
        }
        public static void Register() => _ = registered.Value;
    }
}
