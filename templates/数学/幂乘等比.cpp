
vector<ull> cal_ik(int n, int k)
{
    int i, j, x, c = 0;
    vector<ull> f(n + 1);
    f[0] = k == 0;
    f[1] = 1;
    if (n <= 1) return f;
    vector<int> pr((n / log(n)) * (1 + 1.2762 / log(n)) + 2);
    vector<char> ed(n + 1);
    for (i = 2; i <= n; i++)
    {
        if (!ed[i]) pr[c++] = i, f[i] = ksm(i, k);
        for (j = 0; (x = i * pr[j]) <= n; j++)
        {
            ed[x] = 1;
            f[x] = f[i] * f[pr[j]] % p;
            if (i % pr[j] == 0) break;
        }
    }
    return f;
}
ull sum_n(ull n, ull x, int d)
{
    x %= p;
    if (x == 0) return n > 0 && d == 0;
    vector<ull> id = cal_ik(d + 1, d), y(d + 2);
    ull xp = 1, ix = ksm(x, p - 2);
    int i;
    for (i = 0; i <= d; i++)
    {
        y[i + 1] = (y[i] + xp * id[i]) % p;
        (xp *= x) %= p;
    }
    if (n <= d + 1) return y[n];
    if (x == 1) return interpolation(y, n);
    array<ull, 2> s{0, 0};
    for (i = 0; i <= d + 1; i++)
    {
        (s[d + i & 1] += C(d + 1, i) * xp % p * y[i] % p) %= p;
        (xp *= ix) %= p;
    }
    ull q = (s[1] + p - s[0]) * ksm(ksm((1 + p - x) % p, d + 1), p - 2) % p, ixp = 1;
    y.pop_back();
    for (ull &e : y)
    {
        e = (q + p - e) * ixp % p;
        (ixp *= ix) %= p;
    }
    return (q + (p - ksm(x, n)) * interpolation(y, n)) % p;
}
ull sum_inf(ull x, int d)
{
    if (x == 0) return d == 0;
    assert(x != 1 && x < p);
    ull f, xm1 = ksm(x - 1, p - 2), y = x * xm1 % p;
    f = ksm(y - 1, p - 2) * (ksm(y, d + 1) - 1) % p;
    vector<ull> g = cal_ik(d, d);
    xm1 = p - xm1;
    ull r = 0, k = ksm(y, p - 2), a = ksm(y, d) * (x - 1) % p;
    for (int i = 0; i <= d; i++)
    {
        r = (r + xm1 * f % p * g[i]) % p;
        (xm1 *= p - y) %= p;
        f = (f * (1 + p - x) + C(d + 1, d - i) * a) % p;
        (a *= k) %= p;
    }
    return r;
}
