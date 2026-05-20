template<class info> struct lct
{
	int n;
	vector<info> sum, sum_rev, val, sum_oth;
	vector<int> f, lz;
	vector<array<int, 2>> c;
	lct(int _n, const info &o) :n(_n + 1), sum(n, o), sum_rev(n, o), val(n, o), sum_oth(n, o), f(n), lz(n), c(n) { }
	bool nroot(int x) const
	{
		return c[f[x]][0] == x || c[f[x]][1] == x;
	}
	void pushup(int x)
	{
		sum[x] = val[x];
		sum[x] += sum_oth[x];
		sum_rev[x] = sum[x];
		sum[x] = sum[c[x][0]] + sum[x] + sum[c[x][1]];
		sum_rev[x] = sum_rev[c[x][1]] + sum_rev[x] + sum_rev[c[x][0]];
	}
	void rev(int x)
	{
		if (x)
		{
			swap(c[x][0], c[x][1]);
			swap(sum[x], sum_rev[x]);
			lz[x] ^= 1;
		}
	}
	void pushdown(int x)
	{
		if (lz[x])
		{
			rev(c[x][0]);
			rev(c[x][1]);
			lz[x] = 0;
		}
	}
	void zigzag(int x)
	{
		int y = f[x], z = f[y], typ = (c[y][0] == x);
		if (nroot(y)) c[z][c[z][1] == y] = x;
		f[x] = z; f[y] = x;
		if (c[x][typ]) f[c[x][typ]] = y;
		c[y][typ ^ 1] = c[x][typ]; c[x][typ] = y;
		pushup(y);
	}
	void splay(int x)
	{
		static vector<int> st(n);
		int y, tp;
		st[tp = 1] = y = x;
		while (nroot(y)) st[++tp] = y = f[y];
		while (tp) pushdown(st[tp--]);
		for (; nroot(x); zigzag(x)) if (!nroot(f[x])) continue; else zigzag((c[f[x]][0] == x) ^ (c[f[f[x]]][0] == f[x]) ? x : f[x]);
		pushup(x);
	}
	void access(int x)
	{
		for (int y = 0; x; x = f[y = x])
		{
			splay(x);
			sum_oth[x] -= sum[y];
			sum_oth[x] += sum[c[x][1]];
			c[x][1] = y; pushup(x);
		}
	}
	int findroot(int x)
	{
		access(x); splay(x); pushdown(x);
		while (c[x][0]) pushdown(x = c[x][0]);
		splay(x);
		return x;
	}
	void split(int x, int y)
	{
		makeroot(x);
		access(y);
		splay(y);
	}
	void makeroot(int x)
	{
		access(x);
		splay(x);
		rev(x);
	}
	void link(int x, int y)
	{
		makeroot(x);
		if (x != findroot(y))//可能已经连通
		{
			makeroot(y); f[x] = y;
			sum_oth[y] += sum[x];
			pushup(y);
		}
	}
	void cut(int x, int y)
	{
		makeroot(x);
		if (x == findroot(y))//可能本不连通
		{
			pushdown(x);
			if (c[x][1] == y && !c[y][0] && !c[y][1])//可能连通但无边
			{
				c[x][1] = f[y] = 0;
				pushup(x);
			}
		}
	}
	void set(int x, info y)
	{
		makeroot(x);
		val[x] = y;
		pushup(x);
	}
};
const ull p = 998244353;
struct Q
{
	ull k, b, sum, sz;
	Q operator+(const Q &o) const
	{
		return {k * o.k % p, (b + k * o.b) % p, (sum + k * o.sum + b * o.sz) % p, sz + o.sz};
	}
	void operator+=(const Q &o)
	{
		(sum += k * o.sum + b * o.sz) %= p;
		sz += o.sz;
	}
	void operator-=(const Q &o)
	{
		(sum += p * p * 2 - k * o.sum - b * o.sz) %= p;
		sz -= o.sz;
	}
};
int main()
{
	ios::sync_with_stdio(0); cin.tie(0);
	int n, m, i;
	cin >> n >> m;
	vector<Q> a(n * 2 + 1);
	for (i = 1; i <= n; i++)
	{
		ull x;
		cin >> x;
		a[i] = {1, 0, x, 1};
	}
	vector<pair<int, int>> edges(n);
	for (i = 1; i < n; i++)
	{
		auto &[u, v] = edges[i];
		ull k, b;
		cin >> u >> v >> k >> b;
		++u, ++v;
		a[i + n] = {k, b, 0, 0};
	}
	lct<Q> s(n * 2 - 1, Q{1, 0, 0, 0});
	for (i = 1; i < n * 2; i++) s.set(i, a[i]);
	for (i = 1; i < n; i++)
	{
		auto [u, v] = edges[i];
		s.link(u, n + i);
		s.link(v, n + i);
	}
	while (m--)
	{
		int op;
		cin >> op;
		if (op == 0)
		{
			int u;
			ull x;
			cin >> u >> x;
			++u;
			a[u] = {1, 0, x, 1};
			s.set(u, a[u]);
		}
		else
		{
			int id;
			ull k, b;
			cin >> id >> k >> b;
			++id;
			a[id + n] = {k, b, 0, 0};
			s.set(id + n, a[id + n]);
		}
		int rt;
		cin >> rt;
		++rt;
		s.makeroot(rt);
		cout << s.sum[rt].sum << '\n';
	}
}
