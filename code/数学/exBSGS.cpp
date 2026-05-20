namespace BSGS
{
    template<int N, class X, class Y, ui p = (int)1e6 + 7> struct ht//个数，定义域，值域
    {
        Y a[N], def;
        X v[N];
        int fir[p + 2], nxt[N], st[p + 2];//和模数相适应
        int tp, ds;//自定义模数
        ht(Y def = 0) :def(def) { memset(fir, 0, sizeof fir); tp = ds = 0; }
        Y &operator[](X x)//位置，值
        {
            ui y = x % p;
            for (int i = fir[y]; i; i = nxt[i])
                if (v[i] == x)
                    return a[i];
            v[++ds] = x; a[ds] = def;
            if (!fir[y]) st[++tp] = y;
            nxt[ds] = fir[y]; fir[y] = ds;
            return a[ds];
        }
        Y find(X x)
        {
            ui y = x % p;
            int i;
            for (i = fir[y]; i; i = nxt[i]) if (v[i] == x) return a[i];
            return def;
        }
        void clear()
        {
            ++tp;
            while (--tp) fir[st[tp]] = 0;
            ds = 0;
        }
    };
    const int N = 1e6;
    ht<N, ui, ui> s;
    ull a, B, p, ia;
    ull ksm(ull x, ull y, ull p)
    {
        ull r = 1;
        while (y)
        {
            if (y & 1) (r *= x) %= p;
            (x *= x) %= p; y >>= 1;
        }
        return r;
    }
    void init(ui _a, ui _p, ui _B = 0)
    {
        s.clear();
        B = _B; p = _p; a = _a;
        if (!B) B = sqrt(p) + 2;
        assert(B < N);
        ull x = 1; ia = 0;
        for (int i = 0; i < B; i++)
        {
            ui &y = s[x];
            if (y) return;
            y = i + 1;
            (x *= a) %= p;
        }
        ia = ksm(x, p - 2, p);
    }
    int calc(ull b)
    {
        if (!a) return 1 - min((int)b, 2);
        for (ui i = 0; i * B < p; i++)
        {
            ui x = s.find(b);
            if (x) return i * B + x - 1;
            (b *= ia) %= p;
        }
        return -1;
    }
    int bsgs(ui a, ui b, ui p)
    {
        s.clear();
        a %= p; b %= p;
        if (!a) return 1 - min((int)b, 2);//含 -1
        ui i, k, x, y;
        ull j;
        x = sqrt(p) + 2;
        assert(x < N);
        for (i = 0, j = 1; i < x; i++, (j *= a) %= p)
        {
            if (j == b) return i;
            s[j * b % p] = i + 1;
        }
        k = j;
        for (i = 1; i <= x; i++, (j *= k) %= p)
            if (y = s.find(j))
                return (ull)i * x - y + 1;
        return -1;
    }
    bool isprime(ui p)
    {
        if (p <= 1) return 0;
        for (ui i = 2; i * i <= p; i++) if (p % i == 0) return 0;
        return 1;
    }
    int exgcd(int a, int b)
    {
        if (a == 1) return 1;
        return (1 - (ll)b * exgcd(b % a, a)) / a;//not ull
    }
    int exbsgs(ui a, ui b, ui p)//a^x = b (mod p)
    {
        // if (isprime(p)) return bsgs(a, b, p);
        a %= p; b %= p;
        ui i, k, x;
        int cnt = 0;
        ull j, y = __lg(p);
        for (i = 0, j = 1 % p; i <= y; i++, (j *= a) %= p)
            if (j == b)
                return i;
        y = 1;
        while (1)
        {
            if ((x = gcd(a, p)) == 1) break;
            if (b % x) return -1;//no sol
            ++cnt;
            p /= x; b /= x;
            (y *= a / x) %= p;
        }
        a %= p;
        (b *= (int)p + exgcd(y, p)) % p;
        int r = bsgs(a, b, p);
        return r == -1 ? -1 : r + cnt;
    }
}
using BSGS::bsgs, BSGS::exbsgs, BSGS::init, BSGS::calc;
