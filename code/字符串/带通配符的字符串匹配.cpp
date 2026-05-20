namespace NTT
{
	const int N = 1 << 22;
	const ui p = 998244353, g = 3;
	inline ui ksm(ui x, ui y)
	{
		ui ans = 1;
		while (y)
		{
			if (y & 1) ans = 1llu * ans * x % p;
			y >>= 1; x = 1llu * x * x % p;
		}
		return ans;
	}
	ui r[N], w[N];
	void ntt(vector<ui> &a)
	{
		int n = a.size(), i, j, k;
		for (i = 0; i < n; i++) if (i < r[i]) swap(a[i], a[r[i]]);
		for (k = 1; k < n; k <<= 1)
		{
			for (i = 0; i < n; i += k << 1)
			{
				for (j = 0; j < k; j++)
				{
					ui x = a[i + j], y = 1llu * a[i + j + k] * w[j + k] % p;
					a[i + j] = (x + y) % p; a[i + j + k] = (x + p - y) % p;
				}
			}
		}
	}
	vector<ui> mul(vector <ui> a, vector <ui> b)
	{
		if (a.size() == 0 || b.size() == 0) return { };
		int m = a.size() + b.size() - 1;
		int n = 1 << __lg(m * 2 - 1);
		int i, j, base = __lg(n) - 1;
		ui inv = ksm(n, p - 2);
		for (i = 1; i < n; i++) r[i] = r[i >> 1] >> 1 | (i & 1) << base;
		for (j = 1; j < n; j <<= 1)
		{
			ui wn = ksm(3, (p - 1) / (j << 1));
			w[j] = 1;
			for (i = 1; i < j; i++) w[j + i] = 1llu * w[j + i - 1] * wn % p;
		}
		a.resize(n); b.resize(n);
		ntt(a); ntt(b);
		for (i = 0; i < n; i++) a[i] = 1llu * a[i] * b[i] % p;
		ntt(a); reverse(1 + all(a)); a.resize(n = m);
		for (i = 0; i < n; i++) a[i] = 1llu * a[i] * inv % p;
		return a;
	}
}
vector<int> match(const string &s, const string &t)
{
	using NTT::p, NTT::mul;
	static mt19937 rnd(chrono::steady_clock::now().time_since_epoch().count());
	static array<ui, 256> c;
	static bool inited = 0;
	if (!inited)
	{
		inited = 1;
		for (ui &x : c) x = rnd() % NTT::p;
		c['*'] = 0;//通配符
	}
	int n = s.size(), m = t.size(), i, j;
	if (n < m) return { };
	vector<int> ans;
	vector<ui> f(n), ff(n), fff(n), g(m), gg(m), ggg(m);
	for (i = 0; i < n; i++)
	{
		f[i] = c[s[i]];
		ff[i] = 1llu * f[i] * f[i] % p;
		fff[i] = 1llu * ff[i] * f[i] % p;
	}
	for (i = 0; i < m; i++)
	{
		g[i] = c[t[m - i - 1]];
		gg[i] = 1llu * g[i] * g[i] % p;
		ggg[i] = 1llu * gg[i] * g[i] % p;
	}
	auto fffg = mul(fff, g), ffgg = mul(ff, gg), fggg = mul(f, ggg);
	for (i = 0; i <= n - m; i++) if ((fffg[m - 1 + i] + fggg[m - 1 + i] + 2 * (NTT::p - ffgg[m - 1 + i])) % NTT::p == 0) ans.push_back(i);
	return ans;
}


