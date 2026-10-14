# Math reference

Automatic lambda loops support the .NET 8 `Math`/`MathF` numeric overloads below, including in helpers and value constructors. Argument and result types are the same unless an arrow shows otherwise. Overloads using `decimal`, `nint`, or `nuint` are unsupported.

| Methods | `Math` | `MathF` |
| --- | --- | --- |
| `Abs` | `sbyte`, `short`, `int`, `long`, `float`, `double` | `float` |
| `Min`, `Max`, `Clamp` | `sbyte`, `byte`, `short`, `ushort`, `int`, `uint`, `long`, `ulong`, `float`, `double` | `Min`/`Max` for `float` |
| `Sqrt`, `Cbrt`, `Pow`, `Exp`, `Log`, `Log2`, `Log10` | `double` | `float` |
| `Sin`, `Cos`, `Tan`, `Asin`, `Acos`, `Atan`, `Atan2` | `double` | `float` |
| `Sinh`, `Cosh`, `Tanh`, `Asinh`, `Acosh`, `Atanh` | `double` | `float` |
| `Floor`, `Ceiling`, `Truncate`, `Round` | `double` | `float` |
| `CopySign`, `IEEERemainder`, `FusedMultiplyAdd`, `MinMagnitude`, `MaxMagnitude` | `double` | `float` |
| `BitIncrement`, `BitDecrement`, `ReciprocalEstimate`, `ReciprocalSqrtEstimate` | `double` | `float` |
| `SinCos` | `double` → `(double Sin, double Cos)` | `float` → `(float Sin, float Cos)` |
| `ScaleB` | `(double value, int exponent)` → `double` | `(float value, int exponent)` → `float` |
| `ILogB` | `double` → `int` | `float` → `int` |
| `Sign` | `sbyte`, `short`, `int`, `long`, `float`, `double` → `int` | `float` → `int` |
| `BigMul` | `(int, int)` → `long`; `(long, long, out long low)` → `long` high; `(ulong, ulong, out ulong low)` → `ulong` high | — |
| `DivRem` | Eight integer types → matching `(Quotient, Remainder)` tuple; `int`/`long` also support `out` remainder | — |

`Log` accepts one argument (natural logarithm) or two (explicit base). `Round` supports all four overloads, including digits and/or any of the five `MidpointRounding` modes. Digits must be 0–15 for `double` and 0–6 for `float`.

Floating-point `Min`/`Max` follow .NET NaN and signed-zero rules; `Clamp` uses ordered comparisons. `CopySign` preserves magnitude bits, including NaN payloads. `%` and `IEEERemainder` use their distinct quotient rules, including for exceptional values and subnormals. Rounding preserves signed zero.

`FusedMultiplyAdd` uses portable software arithmetic to round only once across backends and can cost more than native arithmetic. Transcendental results and reciprocal estimates may differ from CLR execution; compare with a numerical tolerance. Parallel reductions may also differ because of grouping; see [reductions](guide.md#reductions-and-atomics).

For execution restrictions and error handling, see the [API guide](guide.md#execution-rules).
