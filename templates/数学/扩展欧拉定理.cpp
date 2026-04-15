namespace Prime
{
    using ui = unsigned;
    const int N = 1e6 + 2;
    const auto M = (N - 1ll) * (N - 1);
    ui pr[N], mn[N], phi[N], cnt;
    int mu[N];
    void init_prime()
    {
        ui i, j, k;
        phi[1] = mu[1] = 1;
        for (i = 2; i < N; i++)
        {
            if (!mn[i])
            {
                pr[cnt++] = i;
                phi[i] = i - 1; mu[i] = -1;
                mn[i] = i;
            }
            for (j = 0; (k = i * pr[j]) < N; j++)
            {
                mn[k] = pr[j];
                if (i % pr[j] == 0)
                {
                    phi[k] = phi[i] * pr[j];
                    break;
                }
                phi[k] = phi[i] * (pr[j] - 1);
                mu[k] = -mu[i];
            }
        }
        //for (i=2;i<N;i++) if (mu[i]<0) mu[i]+=p;
    }
    template<typename T> T getphi(T x)
    {
        assert(M >= x);
        T r = x;
        for (ui i = 0; i < cnt && (T)pr[i] * pr[i] <= x && x >= N; i++) if (x % pr[i] == 0)
        {
            ui y = pr[i], tmp;
            x /= y;
            while (x == (tmp = x / y) * y) x = tmp;
            r = r / y * (y - 1);
        }
        if (x >= N) return r / x * (x - 1);
        while (x > 1)
        {
            ui y = mn[x], tmp;
            x /= y;
            while (x == (tmp = x / y) * y) x = tmp;
            r = r / y * (y - 1);
        }
        return r;
    }
    template<typename T> vector<pair<T, ui>> getw(T x)
    {
        assert(M >= x);
        vector<pair<T, ui>> r;
        for (ui i = 0; i < cnt && (T)pr[i] * pr[i] <= x && x >= N; i++) if (x % pr[i] == 0)
        {
            ui y = pr[i], z = 1, tmp;
            x /= y;
            while (x == (tmp = x / y) * y) x = tmp, ++z;
            r.push_back({y, z});
        }
        if (x >= N)
        {
            r.push_back({x, 1});
            return r;
        }
        while (x > 1)
        {
            ui y = mn[x], z = 1, tmp;
            x /= y;
            while (x == (tmp = x / y) * y) x = tmp, ++z;
            r.push_back({y, z});
        }
        return r;
    }
    int _ = (init_prime(), 0);
}
using Prime::pr, Prime::phi, Prime::getw;
using Prime::mu, Prime::getphi;
ll ksm(ll x, ll y, ll p)
{
    x = (x - p) % p + p;
    ll r = 1;
    while (y)
    {
        if (y & 1) r = (r * x - p) % p + p;
        x = (x * x - p) % p + p;
        y >>= 1;
    }
    return r;
}
struct Q
{
    vector<ll> p;
    Q(ll mod) :p{mod}
    {
        while (p.back() > 1) p.push_back(getphi(p.back()));
    }
    ll operator()(ll a, ll b)
    {
        assert(b);
        if (!a) return (b + 1 & 1) % p[0];
        ll r = 1, i = min<ll>(b, p.size());
        while ((--i) >= 0) r = ksm(a, r, p[i]);
        return r % p[0];
    }
    ll operator()(vector<ll> a)
    {
        assert(a.size());
        ll r = 1, i = 0, j;
        while (i < a.size() && i < p.size() && a[i]) ++i;
        if (i < a.size() && i < p.size())
        {
            j = i;
            while (j < a.size() && !a[j]) ++j;
            a[i] = j - i - 1 & 1;
            ++i;
        }
        while ((--i) >= 0) r = ksm(a[i], r, p[i]);
        return r % p[0];
    }
};
