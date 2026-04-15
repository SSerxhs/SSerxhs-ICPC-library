#define tmpl template<class T>
using ll = long long;
tmpl struct P
{
	ll x, y;
	T v;
};
tmpl struct Q
{
	ll x[2], y[2];
	bool t;
	T s;
	Q() { }
	Q(const P<T> &a)
	{
		x[0] = x[1] = a.x;
		y[0] = y[1] = a.y;
		s = a.v;
	}
};
tmpl bool cmp0(const P<T> &a, const P<T> &b) { return a.x < b.x; }
tmpl bool cmp1(const P<T> &a, const P<T> &b) { return a.y < b.y; }
tmpl struct kdt
{
	vector<P<T>> c;
	vector<Q<T>> a;
	ll m, u, d, l, r;
	T ans;
	bool fir;
	void build(int x, P<T> *b, int n)
	{
		if (x == 1)
		{
			a.resize(m = n << 1);
			a[x].t = 0;
			c.resize(n);
			for (int i = 0; i < n; i++) c[i] = b[i];
		}
		if (n == 1)
		{
			a[x] = Q<T>(b[0]);
			return;
		}
		int mid = n >> 1, c = x << 1;
		nth_element(b, b + mid, b + n, a[x].t ? cmp1<T> : cmp0<T>);
		a[c].t = a[c | 1].t = a[x].t ^ 1;
		build(c, b, mid);
		build(c | 1, b + mid, n - mid);
		a[x].s = a[c].s + a[c | 1].s;
		a[x].x[0] = min(a[c].x[0], a[c | 1].x[0]);
		a[x].x[1] = max(a[c].x[1], a[c | 1].x[1]);
		a[x].y[0] = min(a[c].y[0], a[c | 1].y[0]);
		a[x].y[1] = max(a[c].y[1], a[c | 1].y[1]);
	}
	void find(int x)
	{
		if (x >= m || a[x].x[1]<u || a[x].x[0]>d || a[x].y[1]<l || a[x].y[0]>r) return;
		if (u <= a[x].x[0] && a[x].x[1] <= d && l <= a[x].y[0] && a[x].y[1] <= r)
		{
			ans = fir ? a[x].s : ans + a[x].s;
			fir = 0;
			return;
		}
		find(x << 1); find(x << 1 | 1);
	}
	pair<bool, T> find(ll x1, ll y1, ll x2, ll y2)
	{
		fir = 1;
		ans = { };
		u = x1; d = x2;
		l = y1; r = y2;
		find(1);
		return {!fir, ans};
	}
};
const int N = 2e5 + 2, M = 18;
tmpl struct KDT
{
	kdt<T> s[M];
	P<T> a[N];
	int n, m, i;
	KDT() { n = 0; }
	KDT(int N, ll *x, ll *y, T *w)//[0,n)
	{
		n = N;
		int i, j;
		for (i = 0; i < n; i++) a[i] = {x[i], y[i], w[i]};
		for (i = j = 0; n >> i; i++) if (n >> i & 1) s[i].build(1, a + j, 1 << i), j += 1 << i;
	}
	void insert(ll x, ll y, T w)//插入 (x,y) 的一个数 w
	{
		a[0] = {x, y, w}; m = 1;
		for (i = 0; n & 1 << i; i++) for (auto u : s[i].c) a[m++] = u;
		s[i].build(1, a, m);
		++n;
	}
	pair<bool, T> ask(ll x, ll y, ll xx, ll yy)//查询 [x,xx]*[y,yy] 的和
	{
		T ans;
		bool fir = 1;
		for (i = 0; 1 << i <= n; i++) if (1 << i & n)
		{
			auto [_, tmp] = s[i].find(x, y, xx, yy);
			if (!_) continue;
			ans = fir ? tmp : ans + tmp;
			fir = 0;
		}
		return {!fir, ans};
	}
};
int x[N], y[N], w[N];
int main()
{
	ios::sync_with_stdio(0); cin.tie(0); cout.tie(0);
	int n, q, i;
	cin >> n >> q;
	for (i = 0; i < n; i++) cin >> x[i] >> y[i] >> w[i];
	KDT<ll> s(n, x, y, w);
	while (q--)
	{
		int op, x, y, w;
		cin >> op >> x >> y >> w;
		if (op == 0) s.insert(x, y, w); else
		{
			cin >> op;
			cout << s.ask(x, y, w - 1, op - 1) << '\n';
		}
	}
	return 0;
}
