template<class T, class info> struct range_sort
{
	enum type { inc, dec };
	int n;
	T pl, pr;
	vector<int> root, lc, rc, sz;
	vector<info> s, rs;
	struct treap
	{
		int n, rt;
		vector<ui> pr;
		vector<int> sz, num, lc, rc, lz, rev;
		vector<info> v, rv, s, rs;
		void reverse(int x)
		{
			lz[x] ^= 1;
			rev[x] ^= 1;
			swap(lc[x], rc[x]);
			swap(s[x], rs[x]);
			swap(v[x], rv[x]);
		}
		void pushdown(int x)
		{
			if (lz[x])
			{
				if (lc[x]) reverse(lc[x]);
				if (rc[x]) reverse(rc[x]);
				lz[x] = 0;
			}
		}
		void pushup(int x)
		{
			sz[x] = sz[lc[x]] + sz[rc[x]] + num[x];
			s[x] = v[x];
			rs[x] = rv[x];
			if (lc[x])
			{
				s[x] = s[lc[x]] + s[x];
				rs[x] = rs[x] + rs[lc[x]];
			}
			if (rc[x])
			{
				s[x] = s[x] + s[rc[x]];
				rs[x] = rs[rc[x]] + rs[x];
			}
		}
		int kth;
		void split_kth(int u, int &x, int &y)
		{
			if (!u) return x = y = 0, void();
			pushdown(u);
			if (kth < sz[lc[u]]) split_kth(lc[y = u], x, lc[u]);
			else kth -= sz[lc[u]] + num[u], split_kth(rc[x = u], rc[u], y);
			pushup(u);
		}
		void split_lst(int &x, int &y)
		{
			pushdown(x);
			if (rc[x])
			{
				split_lst(rc[x], y);
				pushup(x);
			}
			else
			{
				y = x;
				x = lc[x];
				lc[y] = 0;
				pushup(y);
			}
		}
		int merge(int x, int y)
		{
			if (!x || !y) return x + y;
			if (pr[x] < pr[y])
			{
				pushdown(x);
				rc[x] = merge(rc[x], y);
				pushup(x);
				return x;
			}
			pushdown(y);
			lc[y] = merge(x, lc[y]);
			pushup(y);
			return y;
		}
		treap(int n) :rt(0), pr(n), sz(n), num(n), lc(n), rc(n), lz(n), v(n), rv(n), s(n), rs(n), rev(n)
		{
			static mt19937 rnd(chrono::steady_clock::now().time_since_epoch().count());
			generate(all(pr), rnd);
		}
		void init(const vector<int> &index, const vector<info> &a)
		{
			n = index.size() - 1;
			for (int i = 1; i <= n; i++)
			{
				s[i] = rs[i] = v[i] = a[index[i]];
				num[i] = sz[i] = 1;
				rt = merge(rt, i);
			}
		}
	};
	treap t;
	int np()
	{
		lc.push_back(0);
		rc.push_back(0);
		sz.push_back(0);
		s.push_back({ });
		rs.push_back({ });
		return lc.size() - 1;
	}
	void pushup(int x)
	{
		if (!x) return;
		sz[x] = sz[lc[x]] + sz[rc[x]];
		if (lc[x] && rc[x])
		{
			s[x] = s[lc[x]] + s[rc[x]];
			rs[x] = rs[rc[x]] + rs[lc[x]];
		}
		else if (lc[x]) s[x] = s[lc[x]], rs[x] = rs[lc[x]];
		else if (rc[x]) s[x] = s[rc[x]], rs[x] = rs[rc[x]];
	}
	void insert(int x, T l, T r, T p, const info &v)
	{
		if (l + 1 == r)
		{
			s[x] = rs[x] = v;
			sz[x] = 1;
			return;
		}
		T mid = midpoint(l, r);
		if (p < mid)
		{
			if (!lc[x]) lc[x] = np();
			insert(lc[x], l, mid, p, v);
		}
		else
		{
			if (!rc[x]) rc[x] = np();
			insert(rc[x], mid, r, p, v);
		}
		pushup(x);
	}
	range_sort(vector<pair<T, info>> a, T pl, T _pr)
		:n(a.size()), pl(pl), pr(_pr - pl), root(n + 1), t(n + 3) {
		np();
		for (int i = 0; i < n; i++)
		{
			a[i].first -= pl;
			root[i + 1] = np();
			insert(root[i + 1], 0, pr, a[i].first, a[i].second);
		}
		t.init(root, s);
	}
	int merge(int x, int y, T l, T r)
	{
		if (!x || !y) return x + y;
		T mid = midpoint(l, r);
		lc[x] = merge(lc[x], lc[y], l, mid);
		rc[x] = merge(rc[x], rc[y], mid, r);
		pushup(x);
		return x;
	}
	pair<int, int> split(int x, int k, T l, T r)
	{
		if (x == 0) return {0, 0};
		if (l + 1 == r) return {0, x};
		T mid = midpoint(l, r);
		if (k < sz[lc[x]])
		{
			auto [u, v] = split(lc[x], k, l, mid);
			lc[x] = v;
			pushup(x);
			if (!sz[x]) x = 0;
			if (!u) return {0, x};
			int y = np();
			lc[y] = u; pushup(y);
			return {y, x};
		}
		auto [u, v] = split(rc[x], k - sz[lc[x]], mid, r);
		rc[x] = u;
		pushup(x);
		if (!sz[x]) x = 0;
		if (!v) return {x, 0};
		int y = np();
		rc[y] = v; pushup(y);
		return {x, y};
	}
	void set_treap(int i, int u, bool swp)
	{
		root[i] = u;
		if (swp)
		{
			t.s[i] = t.v[i] = rs[u];
			t.rs[i] = t.rv[i] = s[u];
		}
		else
		{
			t.s[i] = t.v[i] = s[u];
			t.rs[i] = t.rv[i] = rs[u];
		}
		t.sz[i] = t.num[i] = sz[u];
		t.lc[i] = t.rc[i] = 0;
		t.rev[i] = swp;
	}
	pair<int, int> find(int i)
	{
		if (i == t.sz[t.rt]) return {t.rt, 0};
		t.kth = i;
		int x, y, z, r1, r2;
		t.split_kth(t.rt, x, z);
		t.split_lst(x, y);
		int k = i - t.sz[x], tot = t.num[y];
		if (t.rev[y] && k)
		{
			k = t.num[y] - k;
			tie(r1, r2) = split(root[y], k, 0, pr);
			if (r1) set_treap(i, r2, 1);
			set_treap(i + 1, r1, 1);
			x = t.merge(x, r2 ? i : 0);
			z = t.merge(i + 1, z);
			return {x, z};
		}
		else
		{
			tie(r1, r2) = split(root[y], k, 0, pr);
			if (r1) set_treap(i, r1, t.rev[y]);
			set_treap(i + 1, r2, t.rev[y]);
			x = t.merge(x, r1 ? i : 0);
			z = t.merge(i + 1, z);
			return {x, z};
		}
	}
	tuple<int, int, int> split_range(int l, int r)
	{
		auto [_, z] = find(r);
		t.rt = _;
		auto [x, y] = find(l);
		return {x, y, z};
	}
	void modify(int i, const pair<T, info> &rhs)
	{
		assert(rhs.first >= pl && rhs.first < pl + pr);
		auto [x, y, z] = split_range(i, i + 1);
		root[y] = np();
		insert(root[y], 0, pr, rhs.first - pl, rhs.second);
		set_treap(y, root[y], 0);
		t.rt = t.merge(t.merge(x, y), z);
	}
	info ask(int l, int r)
	{
		auto [x, y, z] = split_range(l, r);
		info res = t.s[y];
		t.rt = t.merge(t.merge(x, y), z);
		return res;
	}
	void merge_dfs(int x, int y)
	{
		if (!y) return;
		if (x != y) root[x] = merge(root[x], root[y], 0, pr), root[y] = 0;
		merge_dfs(x, t.lc[y]);
		merge_dfs(x, t.rc[y]);
	}
	void sort(int l, int r, type op)
	{
		auto [x, y, z] = split_range(l, r);
		merge_dfs(y, y);
		set_treap(y, root[y], op == dec);
		t.rt = t.merge(t.merge(x, y), z);
	}
};
const ull p = 998244353;
struct info
{
	ull k, b;
	info operator+(const info &rhs) const { return {(k * rhs.k) % p, (rhs.b + rhs.k * b) % p}; }
	bool operator<(const info &rhs) const { return 0; }
	ull operator()(ull x) const { return (k * x + b) % p; }
};
int main()
{
	ios::sync_with_stdio(0); cin.tie(0);
	cout << fixed << setprecision(15);
	int n, m, i;
	cin >> n >> m;
	vector<pair<int, info>> a(n);
	for (auto &[x, y] : a)
	{
		auto &[k, b] = y;
		cin >> x >> k >> b;
	}
	range_sort s(a, 0, (int)1e9 + 1);
	while (m--)
	{
		int op;
		cin >> op;
		if (op == 0)
		{
			int i, p;
			ull k, b;
			cin >> i >> p >> k >> b;
			a[i] = {p, {k, b}};
			s.modify(i, {p, {k, b}});
		}
		else
		{
			int l, r;
			cin >> l >> r;
			if (op == 1)
			{
				ull x;
				cin >> x;
				cout << s.ask(l, r)(x) << '\n';
			}
			else s.sort(l, r, op == 2 ? s.inc : s.dec);
		}
	}
}

