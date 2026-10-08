using System.Reflection;
using ILGPU;

namespace ParallelGpu;

/// <summary>Binds a host value array to unmanaged device storage, preserving native arrays without conversion.</summary>
/// <typeparam name="T">The runtime-validated host element type, including nullable values.</typeparam>
internal abstract class DeviceValueBuffer<T> : IDisposable
{
    /// <summary>Gets the uploaded byte view.</summary>
    internal abstract ArrayView<byte> View { get; }
    /// <summary>Gets the physical device element size.</summary>
    internal abstract int ElementSize { get; }
    /// <summary>Allocates uninitialized storage that direct slice uploads must completely fill before launch.</summary>
    /// <param name="gpu">The executor owning the allocation.</param>
    /// <param name="length">The device element count.</param>
    internal abstract void Allocate(Gpu gpu, int length);
    /// <summary>Uploads elements to an executor, converting only when the host representation differs.</summary>
    /// <param name="gpu">The executor owning the buffer.</param>
    /// <param name="values">The packed host elements.</param>
    internal abstract void Upload(Gpu gpu, T[] values);
    /// <summary>Uploads a consecutive slice into existing device storage.</summary>
    /// <param name="offset">The starting destination element offset.</param>
    /// <param name="values">The host elements to copy.</param>
    internal abstract void CopyFrom(int offset, T[] values);
    /// <summary>Downloads and converts a consecutive element slice.</summary>
    /// <param name="offset">The starting element offset.</param>
    /// <param name="values">The host destination.</param>
    internal abstract void Download(int offset, T[] values);
    /// <inheritdoc/>
    public abstract void Dispose();

    /// <summary>Cached constructor binding, avoiding reflection on repeated loop invocations.</summary>
    private static readonly Func<DeviceValueBuffer<T>> factory = (Func<DeviceValueBuffer<T>>)typeof(DeviceValueBuffer<T>)
        .GetMethod(nameof(CreateTyped), BindingFlags.NonPublic | BindingFlags.Static)!
        .MakeGenericMethod(DeviceValueType.MapType(typeof(T)))
        .CreateDelegate(typeof(Func<DeviceValueBuffer<T>>));
    /// <summary>Creates storage bound to the unmanaged representation of this host element type.</summary>
    /// <returns>A typed buffer owner for native values or generated mirrors.</returns>
    internal static DeviceValueBuffer<T> Create() => factory();

    /// <summary>Binds converted storage to its device element type.</summary>
    /// <typeparam name="TDevice">The unmanaged device representation.</typeparam>
    /// <returns>The storage owner.</returns>
    private static DeviceValueBuffer<T> CreateTyped<TDevice>() where TDevice : unmanaged =>
        new DeviceValueBuffer<T, TDevice>();
}

/// <summary>Owns typed device storage and cached converters when conversion is required.</summary>
/// <typeparam name="THost">The original host element type.</typeparam>
/// <typeparam name="TDevice">The unmanaged device element type.</typeparam>
internal sealed class DeviceValueBuffer<THost, TDevice> : DeviceValueBuffer<THost>
    where TDevice : unmanaged
{
    /// <summary>Whether the original arrays already have the device element representation.</summary>
    private static readonly bool native = typeof(THost) == typeof(TDevice);
    /// <summary>The host-to-device converter, compiled once only for different representations.</summary>
    private static readonly Func<THost, TDevice>? pack = native ? null : DeviceValueType.Converter<THost, TDevice>();
    /// <summary>The device-to-host converter, compiled once only for different representations.</summary>
    private static readonly Func<TDevice, THost>? unpack = native ? null : DeviceValueType.Converter<TDevice, THost>();
    /// <summary>The device allocation owned by this batch.</summary>
    private GpuBuffer<TDevice>? buffer;
    /// <inheritdoc/>
    internal override ArrayView<byte> View => buffer!.View.Cast<byte>();
    /// <inheritdoc/>
    internal override int ElementSize => Interop.SizeOf<TDevice>();
    /// <inheritdoc/>
    internal override void Allocate(Gpu gpu, int length) => buffer = gpu.AllocateUninitialized<TDevice>(length);
    /// <inheritdoc/>
    internal override void Upload(Gpu gpu, THost[] values) => buffer = gpu.Upload(Pack(values));
    /// <inheritdoc/>
    internal override void CopyFrom(int offset, THost[] values) => buffer!.CopyFrom(offset, Pack(values));
    /// <summary>Returns the original native array or a converted mirror array.</summary>
    /// <param name="values">The host values.</param>
    /// <returns>The corresponding device values.</returns>
    private static TDevice[] Pack(THost[] values)
    {
        if (native)
            return (TDevice[])(object)values;
        var converted = new TDevice[values.Length];
        for (int i = 0; i < values.Length; i++)
            converted[i] = pack!(values[i]);
        return converted;
    }
    /// <inheritdoc/>
    internal override void Download(int offset, THost[] values)
    {
        if (native)
        {
            buffer!.CopyTo(offset, (TDevice[])(object)values);
            return;
        }
        var converted = new TDevice[values.Length];
        buffer!.CopyTo(offset, converted);
        for (int i = 0; i < values.Length; i++)
            values[i] = unpack!(converted[i]);
    }
    /// <inheritdoc/>
    public override void Dispose() => buffer?.Dispose();
}
