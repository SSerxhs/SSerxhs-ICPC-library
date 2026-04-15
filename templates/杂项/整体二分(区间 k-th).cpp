struct cz
{
	int x, y, kth, pos, typ;
};
cz q[M], st1[M], st2[M];
int a[N], b[N], d[N], ans[N], s[N];
int n, m, t1, t2, i, j, c, gs;
void add(int x, int y)
{
	for (; x <= n; x += x & -x) s[x] += y;
}
int sum(int x)
{
	int ans = 0;
	for (; x; x -= x & -x) ans += s[x];
	return ans;
}
void ztef(int ql, int qr, int l, int r)
{
	if (ql > qr) return;
	int mid = l + r >> 1, i, midd;
	t1 = t2 = 0;
	if (l == r)
	{
		for (i = ql; i <= qr; i++) if (q[i].typ) ans[q[i].pos] = d[l];
		return;
	}
	for (i = ql; i <= qr; i++) if (q[i].typ)
	{
		midd = sum(q[i].y) - sum(q[i].x - 1);
		if (midd >= q[i].kth) st1[++t1] = q[i]; else
		{
			st2[++t2] = q[i];
			st2[t2].kth -= midd;
		}
	}
	else if (q[i].pos <= mid)
	{
		add(q[i].x, 1);
		st1[++t1] = q[i];
	}
	else st2[++t2] = q[i];
	for (i = 1; i <= t1; i++) if (!st1[i].typ) add(st1[i].x, -1);
	for (i = 1; i <= t1; i++) q[i + ql - 1] = st1[i];
	midd = ql + t1 - 1;
	for (i = 1; i <= t2; i++) q[i + midd] = st2[i];
	ztef(ql, midd, l, mid); ztef(midd + 1, qr, mid + 1, r);
}
int main()
{
	cin >> n >> m;
	for (i = 1; i <= n; i++)
	{
		cin >> a[i];
		b[i] = a[i];
	}
	sort(b + 1, b + n + 1);
	d[gs = 1] = b[1];
	for (i = 2; i <= n; i++) if (b[i] != b[i - 1]) d[++gs] = b[i];
	for (i = 1; i <= n; i++) a[i] = lower_bound(d + 1, d + gs + 1, a[i]) - d;
	for (i = 1; i <= n; i++)
	{
		q[i].x = i; q[i].pos = a[i]; q[i].typ = 0;
	}
	for (i = 1; i <= m; i++)
	{
		cin >> q[i + n].x >> q[i + n].y >> q[i + n].kth;
		q[i + n].pos = i; q[i + n].typ = 1;
	}
	ztef(1, n + m, 1, gs);
	for (i = 1; i <= m; i++) printf("%d\n", ans[i]);
}
