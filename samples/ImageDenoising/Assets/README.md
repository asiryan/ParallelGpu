# Input image

`input.png` is the 704 × 469 image supplied for this sample, downloaded unchanged from this [IPOL run](https://ipolcore.ipol.im/api/core/shared_folder/run/55/C13940ABF665D3205E6566EA90D71406/input_0.png).

SHA-256: `3438f2a66bb0ee4958cfce59d3596fe791d22f4f788c0b801da919a5ad393a41`.

The project copies the image to its output directory, so no download is needed at runtime. Set the input path and denoising parameters at the top of [Program.cs](../Program.cs).

To process an existing noisy PNG, such as RayTracer output, set `inputPath` to its path, set `generateNoise` to `false`, and set `Sigma` to the estimated noise level (for example, `10`).
