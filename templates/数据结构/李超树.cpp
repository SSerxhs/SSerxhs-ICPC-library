struct Q
{
	int x0, y0, dx, dy, id;
	Q() :x0(0), y0(-1), dx(1), dy(0), id(-1) { }//y>=0
	Q(int a, int b, int c, int d, int e) :x0(a), y0(b), dx(c), dy(d), id(e) { }
	bool contains(const int &x) const { return x0 <= x && x <= x0 + dx; }
};
bool cmp(const Q &a, const Q &b, int x)//小心数值爆炸
{
	ll A = ((ll)a.y0 * a.dx + (ll)(x - a.x0) * a.dy) * b.dx, B = ((ll)b.y0 * b.dx + (ll)(x - b.x0) * b.dy) * a.dx;
	if (A != B) return A < B;
	return a.id > b.id;
}
bool cmp2(const Q &a, const Q &b)
{
	if (a.y0 + a.dy != b.y0 + b.dy) return a.y0 + a.dy < b.y0 + b.dy;
	return a.id > b.id;
}
const int inf = 1e9;
int ans;
namespace seg
{
	const int N = 4e4 + 2, M = N * 4;
	Q s[M], X[N];
	int n, z, y;
	void init(int nn) { n = nn; for (int i = 1; i <= n * 4; i++) s[i] = Q(); }
	void insert(int x, int l, int r, Q dt)
	{
		int c = x * 2, m = l + r >> 1;
		if (z <= l && r <= y)
		{
			if (cmp(s[x], dt, m)) swap(s[x], dt);
			if (l == r) return;
			if (cmp(s[x], dt, l)) insert(c, l, m, dt);
			else if (cmp(s[x], dt, r)) insert(c + 1, m + 1, r, dt);
			return;
		}
		if (z <= m) insert(c, l, m, dt);
		if (y > m) insert(c + 1, m + 1, r, dt);
	}
	void insert(const Q &o)
	{
		z = o.x0; y = z + o.dx;
		assert(1 <= z && z <= y && y <= n);
		if (z == y)
		{
			if (cmp2(X[z], o)) X[z] = o;
			return;
		}
		insert(1, 1, n, o);
	}
	Q askmax(int p)
	{
		Q ans = s[1].contains(p) ? s[1] : Q();
		int x = 1, l = 1, r = n, c, m;
		while (l < r)
		{
			c = x * 2, m = l + r >> 1;
			if (p <= m) x = c, r = m; else x = c + 1, l = m + 1;
			if (s[x].contains(p) && cmp(ans, s[x], p)) ans = s[x];
		}
		Q o(X[p].x0, X[p].y0 + X[p].dy, 1, 0, 0);
		return cmp(ans, o, p) ? X[p] : ans;
	}
}
int main()
{
	ios::sync_with_stdio(0); cin.tie(0);
	cout << setiosflags(ios::fixed) << setprecision(15);
	int n = 4e4, m, i;
	seg::init(n);
	cin >> m;
	while (m--)
	{
		int op;
		cin >> op;
		if (op)
		{
			int x[2], y[2];
			cin >> x[0] >> y[0] >> x[1] >> y[1];
			for (int &v : x) v = (v + ans - 1) % 39989 + 1;
			for (int &v : y) v = (v + ans - 1) % inf + 1;
			if (x[0] > x[1] || x[0] == x[1] && y[0] > y[1]) swap(x[0], x[1]), swap(y[0], y[1]);
			static int id;
			seg::insert({x[0], y[0], x[1] - x[0], y[1] - y[0], ++id});
		}
		else
		{
			int x;
			cin >> x;
			x = (x + ans - 1) % 39989 + 1;
			cout << (ans = max(0, seg::askmax(x).id)) << '\n';
		}
	}
}

