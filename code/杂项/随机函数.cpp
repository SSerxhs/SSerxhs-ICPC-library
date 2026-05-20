ull rnd(ull x = 0)
{
    static ull s = chrono::high_resolution_clock::now().time_since_epoch().count() ^ 0x9e3779b97f4a7c15;
    s += 0x60bee2bee120fc15;
    x ^= s ^ 0xa0761d6478bd642f;
    ulll m = (ulll)x * (x ^ 0xe7037ed1a0b428db);
    return (ull)m ^ (ull)(m >> 64);
}
