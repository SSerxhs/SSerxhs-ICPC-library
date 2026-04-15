typedef long long ll;
const int N = 1e5 + 2, M = 1 << 14;
ll f[N], ans[N], ta[N];
int a[N], cnt[M], bel[N], pc[M], st[N];
int n, m, ksiz;
struct Q
{
	int z, y, wz;
	bool operator<(const Q &x) const { return (bel[z] < bel[x.z]) || (bel[z] == bel[x.z]) && ((y < x.y) && (bel[z] & 1) || (y > x.y) && (1 ^ bel[z] & 1)); }
};
Q mq(const int x, const int y, const int z)
{
	Q a;
	a.z = x; a.y = y; a.wz = z;
	return a;
}
Q q[N];
vector<Q> b[N];
int main()
{
	ios::sync_with_stdio(false);
	cin.tie(0);
	int i, j, k, l = 1, r = 0, tp = 0, x, na;
	cin >> n >> m >> k; ksiz = sqrt(n);
	for (i = 1; i <= n; i++) { cin >> a[i]; bel[i] = (i - 1) / ksiz + 1; }
	if (k == 0) st[++tp] = 0;
	for (i = 1; i < 16384; i++)
	{
		if (i & 1) pc[i] = pc[i >> 1] + 1; else pc[i] = pc[i >> 1];
		if (pc[i] == k) st[++tp] = i;
	}
	for (i = 1; i <= n; i++)
	{
		j = tp + 1; f[i] = f[i - 1];
		while (--j) f[i] += cnt[st[j] ^ a[i]];
		++cnt[a[i]];
	}
	for (i = 1; i <= m; i++) { cin >> q[i].z >> q[q[i].wz = i].y; }
	sort(q + 1, q + m + 1);
	for (i = 1; i <= m; i++)
	{
		ans[i] = f[q[i].y] - f[r] + f[q[i].z - 1] - f[l - 1];
		if (k == 0) ans[i] += q[i].z - l;
		if (r < q[i].y)
		{
			b[l - 1].push_back(mq(r + 1, q[i].y, -i));
			r = q[i].y;
		}
		if (l > q[i].z)
		{
			b[r].push_back(mq(q[i].z, l - 1, i));
			l = q[i].z;
		}
		if (r > q[i].y)
		{
			b[l - 1].push_back(mq(q[i].y + 1, r, i));
			r = q[i].y;
		}
		if (l < q[i].z)
		{
			b[r].push_back(mq(l, q[i].z - 1, -i));
			l = q[i].z;
		}
	}
	memset(cnt, 0, sizeof(cnt));
	for (i = 1; i <= n; i++)
	{
		j = tp + 1; x = a[i];
		while (--j) ++cnt[x ^ st[j]];
		for (j = 0; j < b[i].size(); j++)
		{
			na = 0; l = b[i][j].z; r = b[i][j].y;
			for (k = l; k <= r; k++) na += cnt[a[k]];
			if (b[i][j].wz > 0) ans[b[i][j].wz] += na; else ans[-b[i][j].wz] -= na;
		}
	}
	for (i = 2; i <= m; i++) ans[i] += ans[i - 1];
	for (i = 1; i <= m; i++) ta[q[i].wz] = ans[i];
	for (i = 1; i <= m; i++) printf("%lld\n", ta[i]);
}

