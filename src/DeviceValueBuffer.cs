using System.Reflection;
using ILGPU;

namespace ParallelGpu;

/// <summary>Packs a host structure array into a blittable device mirror and converts downloaded values back.</summary>
/// <typeparam name="T">The host element type.</typeparam>
internal abstract class DeviceValueBuffer<T> : IDisposable where T : unmanaged
{
    /// <summary>Gets the uploaded byte view.</summary>
    internal abstract ArrayView<byte> View { get; }
    /// <summary>Gets the physical device element size.</summary>
    internal abstract int ElementSize { get; }
    /// <summary>Uploads converted elements to an executor.</summary>
    /// <param name="gpu">The executor owning the buffer.</param>
    /// <param name="values">The packed host elements.</param>
    internal abstract void Upload(Gpu gpu, T[] values);
    /// <summary>Downloads and converts a consecutive element slice.</summary>
    /// <param name="offset">The starting element offset.</param>
    /// <param name="values">The host destination.</param>
    internal abstract void Download(int offset, T[] values);
    /// <inheritdoc/>
    public abstract void Dispose();

    /// <summary>Creates storage for the generated mirror of this host element type.</summary>
    /// <returns>A typed converter and buffer owner for the mirror.</returns>
    internal static DeviceValueBuffer<T> Create() => (DeviceValueBuffer<T>)typeof(DeviceValueBuffer<T>)
        .GetMethod(nameof(CreateTyped), BindingFlags.NonPublic | BindingFlags.Static)!
        .MakeGenericMethod(DeviceValueType.MapType(typeof(T))).Invoke(null, null)!;

    /// <summary>Binds converted storage to its device element type.</summary>
    /// <typeparam name="TDevice">The public blittable device mirror.</typeparam>
    /// <returns>The converted storage owner.</returns>
    private static DeviceValueBuffer<T> CreateTyped<TDevice>() where TDevice : unmanaged =>
        new DeviceValueBuffer<T, TDevice>();
}

/// <summary>Owns one typed mirror buffer and reusable element converters.</summary>
/// <typeparam name="THost">The original host element type.</typeparam>
/// <typeparam name="TDevice">The device mirror element type.</typeparam>
internal sealed class DeviceValueBuffer<THost, TDevice> : DeviceValueBuffer<THost>
    where THost : unmanaged where TDevice : unmanaged
{
    /// <summary>The host-to-device converter, compiled once for this pair of representations.</summary>
    private static readonly Func<THost, TDevice> pack = DeviceValueType.Converter<THost, TDevice>();
    /// <summary>The device-to-host converter, compiled once for this pair of representations.</summary>
    private static readonly Func<TDevice, THost> unpack = DeviceValueType.Converter<TDevice, THost>();
    /// <summary>The device allocation owned by this batch.</summary>
    private GpuBuffer<TDevice>? buffer;
    /// <inheritdoc/>
    internal override ArrayView<byte> View => buffer!.View.Cast<byte>();
    /// <inheritdoc/>
    internal override int ElementSize => Interop.SizeOf<TDevice>();
    /// <inheritdoc/>
    internal override void Upload(Gpu gpu, THost[] values)
    {
        var converted = new TDevice[values.Length];
        for (int i = 0; i < values.Length; i++)
            converted[i] = pack(values[i]);
        buffer = gpu.Upload(converted);
    }
    /// <inheritdoc/>
    internal override void Download(int offset, THost[] values)
    {
        var converted = new TDevice[values.Length];
        buffer!.CopyTo(offset, converted);
        for (int i = 0; i < values.Length; i++)
            values[i] = unpack(converted[i]);
    }
    /// <inheritdoc/>
    public override void Dispose() => buffer?.Dispose();
}
