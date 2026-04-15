struct Q
{
	int k;
	ll b;
	ll y(const int &x) const { return (ll)k * x + b; }
};
const int inf = 1e9;
const ll INF = 1e18;
struct seg//可以析构，不能并行
{
	const static int N = 4e5 + 2, M = N * 8 * 8 + (1 << 23);
	const static ll npos = 9e18;
	static Q s[M];
	static int c[M][2], id;
	int z, y, L, R;
	seg(int l, int r)
	{
		L = l; R = r; id = 1;
		s[1] = {0, npos};
		assert(L <= R && (ll)R - L < 1ll << 32);
	}
private:
	void insert(int &x, int l, int r, Q o)
	{
		if (!x)
		{
			x = ++id;
			assert(id < M);
			s[x] = {0, npos};
		}
		int m = l + (r - l >> 1);
		if (z <= l && r <= y)
		{
			if (s[x].y(m) > o.y(m)) swap(s[x], o);
			if (s[x].y(l) > o.y(l)) insert(c[x][0], l, m, o);
			else if (s[x].y(r) > o.y(r)) insert(c[x][1], m + 1, r, o);
			return;
		}
		if (z <= m) insert(c[x][0], l, m, o);
		if (y > m) insert(c[x][1], m + 1, r, o);
	}
public:
	void insert(const Q &x, const int &l, const int &r)//[l,r]
	{
		z = l; y = r; int tmp = 1;
		insert(tmp, L, R, x);
		assert(tmp == 1);
	}
	ll askmin(const int &p)
	{
		ll res = s[1].y(p);
		int l = L, r = R, m, x = 1;
		while (l < r)
		{
			m = l + (r - l >> 1);
			if (p <= m) x = c[x][0], r = m; else x = c[x][1], l = m + 1;
			if (!x) return res;
			res = min(res, s[x].y(p));
		}
		return res;
	}
	~seg()
	{
		++id;
		while (--id) c[id][0] = c[id][1] = 0;
	}
};
Q seg::s[seg::M];
int seg::c[seg::M][2], seg::id;

