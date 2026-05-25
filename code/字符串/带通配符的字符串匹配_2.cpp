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
void ntt(vector <ui> &a)
{
	int n = a.size(), i, j, k;
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
vector<int> match(string s, string t, char ch = '*')// 单次程序运行中中通配符不可修改
{
	static mt19937 rnd(chrono::steady_clock::now().time_since_epoch().count());
	static array<ui, 256> c;
	static bool inited = 0;
	if (!inited)
	{
		inited = 1;
		for (ui &x : c) x = rnd() % p;
		// for (int i=0; i<256; i++) c[i]=i-96;
		c[ch] = 0;//通配符
	}
	int n = s.size(), m = t.size(), i, j;
	if (n < m) return { };
	vector<int> ans;
	int N = 1 << __lg(n * 2 - 1), base = __lg(N) - 1;
	vector<ui> f(N), ff(N), fff(N), g(N), gg(N), ggg(N);
	reverse(all(t));
	s.resize(N, ch), t.resize(N, ch);
	for (i = 0; i < N; i++)
	{
		r[i] = r[i >> 1] >> 1 | (i & 1) << base;
		if (i < r[i])
		{
			swap(s[i], s[r[i]]);
			swap(t[i], t[r[i]]);
		}
	}
	for (j = 1; j < N; j <<= 1)
	{
		ui wn = ksm(3, (p - 1) / (j << 1));
		w[j] = 1;
		for (i = 1; i < j; i++) w[j + i] = 1llu * w[j + i - 1] * wn % p;
	}
	for (i = 0; i < N; i++)
	{
		f[i] = c[s[i]];
		ff[i] = 1llu * f[i] * f[i] % p;
		fff[i] = 1llu * ff[i] * f[i] % p;
		g[i] = c[t[i]];
		gg[i] = 1llu * g[i] * g[i] % p;
		ggg[i] = 1llu * gg[i] * g[i] % p;
	}
	ntt(f); ntt(ff); ntt(fff); ntt(g); ntt(gg); ntt(ggg);
	for (i = 0; i < N; i++) f[i] = (1llu * fff[i] * g[i] + 1llu * f[i] * ggg[i] + 2llu * (p - ff[i]) * gg[i]) % p;
	for (i = 0; i < N; i++) if (i < r[i]) swap(f[i], f[r[i]]);
	ntt(f); reverse(1 + all(f));
	for (i = 0; i <= n - m; i++) if (f[m + i - 1] == 0) ans.push_back(i);
	return ans;
}

