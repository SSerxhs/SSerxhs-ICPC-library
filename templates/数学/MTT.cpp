namespace MTT
{
	template<ull p> constexpr ull ksm(ull x, ull y = p - 2)
	{
		ull r = 1;
		while (y)
		{
			if (y & 1) r = r * x % p;
			x = x * x % p;
			y >>= 1;
		}
		return r;
	}
	int cal(int x) { return 1 << __lg(max(x, 1) * 2 - 1); }
	const int N = 1 << 22;
	const ull p = 1e9 + 7, g = 3,
		p1 = 7 << 26 | 1, p2 = 119 << 23 | 1, p3 = 479 << 21 | 1,//三模，原根都是 3，非常好
		inv_p1 = ksm<p2>(p1), inv_p12 = ksm<p3>(p1 * p2 % p3), _p12 = p1 * p2 % p;//三模，1 关于 2 逆，1*2 关于 3 逆，1*2 mod 3
	int r[N];
	struct P
	{
		ull v1, v2, v3;
		P operator+(const P &o) const { return {v1 + o.v1, v2 + o.v2, v3 + o.v3}; }
		P operator-(const P &o) const { return {v1 + p1 - o.v1, v2 + p2 - o.v2, v3 + p3 - o.v3}; }
		P operator*(const P &o) const { return {v1 * o.v1, v2 * o.v2, v3 * o.v3}; }
		void operator+=(const P &o) { v1 += o.v1, v2 += o.v2, v3 += o.v3; }
		void operator-=(const P &o) { v1 += p1 - o.v1, v2 += p2 - o.v2, v3 += p3 - o.v3; }
		void operator*=(const P &o) { v1 *= o.v1, v2 *= o.v2, v3 *= o.v3; }
		void mod() { v1 %= p1, v2 %= p2, v3 %= p3; }
	};
	P w[N];
	void init(int n)
	{
		static int pr = 0, pw = 0;
		if (pr == n) return;
		int b = __lg(n) - 1, i, j, k;
		for (i = 1; i < n; i++) r[i] = r[i >> 1] >> 1 | (i & 1) << b;
		if (pw < n)
		{
			for (j = 1; j < n; j = k)
			{
				k = j * 2;
				P wn = {ksm<p1>(g, (p1 - 1) / k), ksm<p2>(g, (p2 - 1) / k), ksm<p3>(g, (p3 - 1) / k)};
				w[j] = {1, 1, 1};
				for (i = j + 1; i < k; i++) w[i] = w[i - 1] * wn, w[i].mod();
			}
			pw = n;
		}
		pr = n;
	}
	void dft(vector<P> &a, int o = 0)
	{
		int n = a.size(), i, j, k;
		P *f, *g, *wn, *b = a.data(), x, y;
		init(n);
		for (i = 1; i < n; i++) if (i < r[i]) swap(a[i], a[r[i]]);
		for (k = 1; k < n; k *= 2)
		{
			wn = w + k;
			for (i = 0; i < n; i += k * 2)
			{
				f = b + i; g = b + i + k;
				for (j = 0; j < k; j++)
				{
					y = g[j] * wn[j];
					y.mod();
					g[j] = f[j] - y;
					f[j] += y;
				}
			}
			if (k * 2 == n || k == 1 << 14) for (P &x : a) x.mod();
		}
		if (o)
		{
			x = {ksm<p1>(n), ksm<p2>(n), ksm<p3>(n)};
			for (P &y : a) y *= x, y.mod();
			reverse(1 + all(a));
		}
	}
	struct Q :vector<ull>
	{
		Q(int x = 1) :vector(x) { }
		Q &operator%=(int n) { resize(n); return *this; }
	};
	Q &operator*=(Q &f, const Q &g)
	{
		int n = f.size() + g.size() - 1, m = cal(n), i;
		vector<P> F(m, {0, 0, 0}), G(m, {0, 0, 0});
		for (i = 0; i < f.size(); i++) F[i] = {f[i] % p1, f[i] % p2, f[i] % p3};
		for (i = 0; i < g.size(); i++) G[i] = {g[i] % p1, g[i] % p2, g[i] % p3};
		dft(F); dft(G);
		for (i = 0; i < m; i++) F[i] *= G[i], F[i].mod();
		dft(F, 1);
		f %= n;
		ull x;
		for (i = 0; i < n; i++)
		{
			auto [r1, r2, r3] = F[i];
			x = (r2 + p2 - r1) * inv_p1 % p2 * p1 + r1;
			f[i] = ((x + p3 - r3) % p3 * (p3 - inv_p12) % p3 * _p12 + x) % p;
		}
		return f;
	}//5e5 440ms
	Q operator*(Q f, const Q &g) { return f *= g; }
}
using MTT::p;
using poly = MTT::Q;

