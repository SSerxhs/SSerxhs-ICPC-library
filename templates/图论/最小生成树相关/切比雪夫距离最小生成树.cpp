const int N = 3e5 + 2, M = N << 2;
struct P
{
	int u, v, w;
	P(int a = 0, int b = 0, int c = 0) :u(a), v(b), w(c) { }
	bool operator<(const P &o) const { return w < o.w; }
};
struct Q
{
	int x, y, id;
	Q(int a = 0, int b = 0, int c = 0) :x(a), y(b), id(c) { }
	bool operator<(const Q &o) const { return x != o.x ? x > o.x : y > o.y; }
};
ll ans;
P lb[M];
Q a[N], b[N];
int f[N], c[N];
int n, m, i, x, y;
struct bit
{
	int a[N], pos[N], n;
	void init(int &nn)
	{
		memset(a + 1, 0x7f, (n = nn) * sizeof a[0]);
		memset(pos + 1, 0, n * sizeof pos[0]);
	}
	void mdf(int x, const int y, const int z)
	{
		if (a[x] > y) a[x] = y, pos[x] = z;
		while (x -= x & -x) if (a[x] > y) a[x] = y, pos[x] = z;
	}
	int sum(int x)
	{
		int r = a[x], rr = pos[x];
		while ((x += x & -x) <= n) if (a[x] < r) r = a[x], rr = pos[x];
		return rr;
	}
};
bit s;
void cal()
{
	int i, x, y;
	s.init(n);
	memcpy(b + 1, a + 1, sizeof(Q) * n);
	sort(a + 1, a + n + 1);
	for (i = 1; i <= n; i++) c[i] = a[i].y - a[i].x;
	sort(c + 1, c + n + 1);
	for (i = 1; i <= n; i++)
	{
		if (x = s.sum(y = lower_bound(c + 1, c + n + 1, a[i].y - a[i].x) - c))
			lb[++m] = P(a[x].id, a[i].id, a[x].x + a[x].y - a[i].x - a[i].y);//谨防 int 爆
		s.mdf(y, a[i].y + a[i].x, i);
	}
	memcpy(a + 1, b + 1, sizeof(Q) * n);
}
int getf(int x) { return f[x] == x ? x : f[x] = getf(f[x]); }
int main()
{
	cin >> n;
	for (i = 1; i <= n; i++) {
		cin >> a[f[i] = a[i].id = i].x >> a[i].y;
		swap(a[i].x, a[i].y); a[i] = Q(a[i].x + a[i].y, a[i].x - a[i].y, i);
	}
	cal(); for (i = 1; i <= n; i++) swap(a[i].x, a[i].y);
	cal(); for (i = 1; i <= n; i++) a[i].y = -a[i].y;
	cal(); for (i = 1; i <= n; i++) swap(a[i].x, a[i].y);
	cal(); sort(lb + 1, lb + m + 1);
	for (i = 1; i <= m; i++) if ((x = getf(lb[i].u)) != (y = getf(lb[i].v))) f[x] = y, ans += lb[i].w;
	cout << ans / 2 << endl;
}

