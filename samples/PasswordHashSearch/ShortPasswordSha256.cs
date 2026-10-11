namespace PasswordHashSearch;

/// <summary>The eight big-endian words of a SHA-256 digest, with no managed fields.</summary>
public struct Sha256Digest
{
    /// <summary>The first 32-bit digest word.</summary>
    public uint A;

    /// <summary>The second 32-bit digest word.</summary>
    public uint B;

    /// <summary>The third 32-bit digest word.</summary>
    public uint C;

    /// <summary>The fourth 32-bit digest word.</summary>
    public uint D;

    /// <summary>The fifth 32-bit digest word.</summary>
    public uint E;

    /// <summary>The sixth 32-bit digest word.</summary>
    public uint F;

    /// <summary>The seventh 32-bit digest word.</summary>
    public uint G;

    /// <summary>The eighth 32-bit digest word.</summary>
    public uint H;

    /// <summary>Compares all eight words of two SHA-256 digests.</summary>
    /// <param name="other">The digest to compare.</param>
    /// <returns>True when every digest word matches.</returns>
    public readonly bool EqualsDigest(Sha256Digest other) =>
        A == other.A && B == other.B && C == other.C && D == other.D &&
        E == other.E && F == other.F && G == other.G && H == other.H;
}

/// <summary>Device-compatible SHA-256 specialized for passwords of one to eight ASCII bytes.</summary>
public static class ShortPasswordSha256
{
    /// <summary>The maximum password length that fits in the packed 64-bit input.</summary>
    public const int MaximumPasswordLength = 8;

    /// <summary>Hashes a short ASCII password using one SHA-256 block without managed allocations.</summary>
    /// <param name="packedAscii">The password bytes, with the first character in the least significant byte.</param>
    /// <param name="length">The password length from one through eight, validated by the caller.</param>
    /// <returns>The SHA-256 digest of the requested bytes; unused upper bytes are ignored.</returns>
    public static Sha256Digest Compute(ulong packedAscii, int length)
    {
        // SHA-256 additions wrap modulo 2^32, including the final digest accumulation.
        unchecked
        {
            // Convert the packed little-endian bytes to SHA-256's big-endian message words.
            uint word0 = (uint)(packedAscii & 0xfful) << 24;
            if (length > 1)
                word0 |= (uint)(packedAscii & 0xff00ul) << 8;
            if (length > 2)
                word0 |= (uint)(packedAscii & 0xff0000ul) >> 8;
            if (length > 3)
                word0 |= (uint)((packedAscii >> 24) & 0xfful);

            uint word1 = 0;
            if (length > 4)
                word1 |= (uint)((packedAscii >> 32) & 0xfful) << 24;
            if (length > 5)
                word1 |= (uint)((packedAscii >> 40) & 0xfful) << 16;
            if (length > 6)
                word1 |= (uint)((packedAscii >> 48) & 0xfful) << 8;
            if (length > 7)
                word1 |= (uint)((packedAscii >> 56) & 0xfful);

            // Append 0x80 after the password; remaining block bytes are zero except the bit length.
            uint word2 = 0;
            if (length < 4)
                word0 |= 0x80000000u >> (length * 8);
            else if (length < 8)
                word1 |= 0x80000000u >> ((length - 4) * 8);
            else
                word2 = 0x80000000u;

            var schedule = new WordSchedule
            {
                W0 = word0,
                W1 = word1,
                W2 = word2,
                W15 = (uint)(length * 8)
            };

            uint a = 0x6a09e667u;
            uint b = 0xbb67ae85u;
            uint c = 0x3c6ef372u;
            uint d = 0xa54ff53au;
            uint e = 0x510e527fu;
            uint f = 0x9b05688cu;
            uint g = 0x1f83d9abu;
            uint h = 0x5be0cd19u;

            for (int round = 0; round < 64; round++)
            {
                uint word = schedule.Take(round);
                uint choose = (e & f) ^ (~e & g);
                uint majority = (a & b) ^ (a & c) ^ (b & c);
                uint sum1 = RotateRight(e, 6) ^ RotateRight(e, 11) ^ RotateRight(e, 25);
                uint sum0 = RotateRight(a, 2) ^ RotateRight(a, 13) ^ RotateRight(a, 22);
                uint t1 = h + sum1 + choose + RoundConstant(round) + word;
                uint t2 = sum0 + majority;

                h = g;
                g = f;
                f = e;
                e = d + t1;
                d = c;
                c = b;
                b = a;
                a = t1 + t2;
            }

            return new Sha256Digest
            {
                A = a + 0x6a09e667u,
                B = b + 0xbb67ae85u,
                C = c + 0x3c6ef372u,
                D = d + 0xa54ff53au,
                E = e + 0x510e527fu,
                F = f + 0x9b05688cu,
                G = g + 0x1f83d9abu,
                H = h + 0x5be0cd19u
            };
        }
    }

    /// <summary>Rotates a 32-bit word right, wrapping shifted bits back to the left.</summary>
    /// <param name="value">The word to rotate.</param>
    /// <param name="bits">The rotation count from one through 31.</param>
    /// <returns>The rotated word.</returns>
    private static uint RotateRight(uint value, int bits) =>
        (value >> bits) | (value << (32 - bits));

    /// <summary>Computes the first small sigma function used by the SHA-256 message schedule.</summary>
    /// <param name="value">The schedule word to transform.</param>
    /// <returns>The XOR of two rotations and a logical right shift.</returns>
    private static uint SmallSigma0(uint value) =>
        RotateRight(value, 7) ^ RotateRight(value, 18) ^ (value >> 3);

    /// <summary>Computes the second small sigma function used by the SHA-256 message schedule.</summary>
    /// <param name="value">The schedule word to transform.</param>
    /// <returns>The XOR of two rotations and a logical right shift.</returns>
    private static uint SmallSigma1(uint value) =>
        RotateRight(value, 17) ^ RotateRight(value, 19) ^ (value >> 10);

    /// <summary>Stores a rolling 16-word message schedule without allocating an array per GPU thread.</summary>
    private struct WordSchedule
    {
        /// <summary>The first eight words of the current schedule window.</summary>
        public uint W0, W1, W2, W3, W4, W5, W6, W7;

        /// <summary>The last eight words of the current schedule window.</summary>
        public uint W8, W9, W10, W11, W12, W13, W14, W15;

        /// <summary>Returns the current round's word and advances the schedule window.</summary>
        /// <param name="round">The compression round from zero through 63, supplied in ascending order.</param>
        /// <returns>The message word used by this compression round.</returns>
        public uint Take(int round)
        {
            uint word = W0;
            uint next = 0;
            // At round r, the window holds W[r] through W[r+15]; only expand through W[63].
            if (round < 48)
                next = unchecked(SmallSigma1(W14) + W9 + SmallSigma0(W1) + W0);

            W0 = W1;
            W1 = W2;
            W2 = W3;
            W3 = W4;
            W4 = W5;
            W5 = W6;
            W6 = W7;
            W7 = W8;
            W8 = W9;
            W9 = W10;
            W10 = W11;
            W11 = W12;
            W12 = W13;
            W13 = W14;
            W14 = W15;
            W15 = next;
            return word;
        }
    }

    /// <summary>Returns a SHA-256 round constant using device-compatible literals.</summary>
    /// <param name="round">The compression round from zero through 63.</param>
    /// <returns>The constant added during this compression round.</returns>
    // Literal switch arms are available on the device without a captured static array.
    private static uint RoundConstant(int round) => round switch
    {
        0 => 0x428a2f98u, 1 => 0x71374491u, 2 => 0xb5c0fbcfu, 3 => 0xe9b5dba5u,
        4 => 0x3956c25bu, 5 => 0x59f111f1u, 6 => 0x923f82a4u, 7 => 0xab1c5ed5u,
        8 => 0xd807aa98u, 9 => 0x12835b01u, 10 => 0x243185beu, 11 => 0x550c7dc3u,
        12 => 0x72be5d74u, 13 => 0x80deb1feu, 14 => 0x9bdc06a7u, 15 => 0xc19bf174u,
        16 => 0xe49b69c1u, 17 => 0xefbe4786u, 18 => 0x0fc19dc6u, 19 => 0x240ca1ccu,
        20 => 0x2de92c6fu, 21 => 0x4a7484aau, 22 => 0x5cb0a9dcu, 23 => 0x76f988dau,
        24 => 0x983e5152u, 25 => 0xa831c66du, 26 => 0xb00327c8u, 27 => 0xbf597fc7u,
        28 => 0xc6e00bf3u, 29 => 0xd5a79147u, 30 => 0x06ca6351u, 31 => 0x14292967u,
        32 => 0x27b70a85u, 33 => 0x2e1b2138u, 34 => 0x4d2c6dfcu, 35 => 0x53380d13u,
        36 => 0x650a7354u, 37 => 0x766a0abbu, 38 => 0x81c2c92eu, 39 => 0x92722c85u,
        40 => 0xa2bfe8a1u, 41 => 0xa81a664bu, 42 => 0xc24b8b70u, 43 => 0xc76c51a3u,
        44 => 0xd192e819u, 45 => 0xd6990624u, 46 => 0xf40e3585u, 47 => 0x106aa070u,
        48 => 0x19a4c116u, 49 => 0x1e376c08u, 50 => 0x2748774cu, 51 => 0x34b0bcb5u,
        52 => 0x391c0cb3u, 53 => 0x4ed8aa4au, 54 => 0x5b9cca4fu, 55 => 0x682e6ff3u,
        56 => 0x748f82eeu, 57 => 0x78a5636fu, 58 => 0x84c87814u, 59 => 0x8cc70208u,
        60 => 0x90befffau, 61 => 0xa4506cebu, 62 => 0xbef9a3f7u, _ => 0xc67178f2u
    };
}
