const ull p = 998244353;
struct Q
{
	int n, m;
	ull w;
	int typ;
	bool operator<(const Q &o) const
	{
		if (n != o.n) return n < o.n;
		return typ < o.typ;
	}
};
template<class T> struct tork
{
	vector<T> a;
	int n;
	tork(const vector<T> &b) :a(all(b))
	{
		sort(all(a));
		a.resize(unique(all(a)) - a.begin());
		n = a.size();
	}
	tork(const T *first, const T *last) :a(first, last)
	{
		sort(all(a));
		a.resize(unique(all(a)) - a.begin());
		n = a.size();
	}
	void get(T &x) { x = lower_bound(all(a), x) - a.begin() + 1; }
	T operator[](const int &x) { return a[x]; }
};
struct bit
{
	vector<ull> a;
	int n;
	bit() { }
	bit(int nn) :n(nn), a(nn + 1) { }
	template<class T> bit(int nn, T *b) : n(nn), a(nn + 1)
	{
		for (int i = 1; i <= n; i++) a[i] = b[i];
		for (int i = 1; i <= n; i++) if (i + (i & -i) <= n) a[i + (i & -i)] += a[i];
	}
	void add(int x, ull y)
	{
		// cerr<<"add "<<x<<" by "<<y<<endl;
		assert(1 <= x && x <= n);
		if ((a[x] += y) >= p) a[x] -= p;
		while ((x += x & -x) <= n) if ((a[x] += y) >= p) a[x] -= p;
	}
	ull sum(int x)
	{
		// cerr<<"sum "<<x;
		assert(0 <= x && x <= n);
		ull r = a[x];
		while (x ^= x & -x) r += a[x];
		// cerr<<"= "<<r<<endl;
		return r % p;
	}
	ull sum(int x, int y)
	{
		return (sum(y) + p - sum(x - 1)) % p;
	}
};
struct matrix
{
	int l, d, r, u;
	ull w;
};
vector<ull> rec_add_rec_sum(const vector<matrix> &op, const vector<matrix> &query)
{
	vector<Q> a[4];
	int n = op.size(), m = query.size(), i;
	for (auto &v : a) v.reserve(n + m << 2);
	for (auto [l, d, r, u, w] : op)//[l,r)*[d,u) += w
	{
		a[0].push_back({l, d, w * l % p * d % p, -1});
		a[1].push_back({l, d, w * l % p, -1});
		a[2].push_back({l, d, w * d % p, -1});
		a[3].push_back({l, d, w, -1});
		w = (p - w) % p;
		a[0].push_back({l, u, w * l % p * u % p, -1});
		a[1].push_back({l, u, w * l % p, -1});
		a[2].push_back({l, u, w * u % p, -1});
		a[3].push_back({l, u, w, -1});
		a[0].push_back({r, d, w * r % p * d % p, -1});
		a[1].push_back({r, d, w * r % p, -1});
		a[2].push_back({r, d, w * d % p, -1});
		a[3].push_back({r, d, w, -1});
		w = (p - w) % p;
		a[0].push_back({r, u, w * r % p * u % p, -1});
		a[1].push_back({r, u, w * r % p, -1});
		a[2].push_back({r, u, w * u % p, -1});
		a[3].push_back({r, u, w, -1});
	}
	i = 0;
	for (auto [l, d, r, u, w] : query)//ask sum of [l,r)*[d,u)
	{
		a[0].push_back({l, d, 1, i});
		a[1].push_back({l, d, (p * 2 - d) % p, i});
		a[2].push_back({l, d, (p * 2 - l) % p, i});
		a[3].push_back({l, d, (ull)l * d % p, i});
		a[0].push_back({l, u, p - 1, i});
		a[1].push_back({l, u, u % p, i});
		a[2].push_back({l, u, l % p, i});
		a[3].push_back({l, u, (p * 2 - l) * u % p, i});
		a[0].push_back({r, u, 1, i});
		a[1].push_back({r, u, (p * 2 - u) % p, i});
		a[2].push_back({r, u, (p * 2 - r) % p, i});
		a[3].push_back({r, u, (ull)u * r % p, i});
		a[0].push_back({r, d, p - 1, i});
		a[1].push_back({r, d, d % p, i});
		a[2].push_back({r, d, r % p, i});
		a[3].push_back({r, d, (p * 2 - d) * r % p, i});
		++i;
	}
	assert(a[0].size() == n + m << 2);
	vector<ull> ans(m);
	auto cal = [&](vector<Q> a) {
		int n = a.size(), i;
		vector<int> b(n);
		for (i = 0; i < n; i++) b[i] = (a[i].m -= a[i].typ >= 0), a[i].n -= a[i].typ >= 0;
		sort(all(a));
		tork t(b);
		for (i = 0; i < n; i++) t.get(a[i].m);
		int m = t.a.size();
		bit s(m);
		for (auto [n, m, w, typ] : a) if (typ >= 0) ans[typ] = (ans[typ] + s.sum(m) * w) % p; else s.add(m, w);
	};
	for (auto &v : a) cal(v);
	return ans;
}
