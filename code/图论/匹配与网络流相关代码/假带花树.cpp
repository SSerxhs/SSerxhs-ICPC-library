mt19937 rnd(3214);
vector<int> lj[N];
int lk[N], ed[N];
int n, m, cnt, i, t, x, y, ans, la;
bool dfs(int x)
{
	ed[x] = cnt; int v;
	shuffle(lj[x].begin(), lj[x].end(), rnd);
	for (auto u : lj[x]) if (ed[v = lk[u]] != cnt)
	{
		lk[v] = 0, lk[u] = x, lk[x] = u;
		if (!v || dfs(v)) return 1;
		lk[v] = u, lk[u] = v, lk[x] = 0;
	}
	return 0;
}
int main()
{
	srand(time(0)); la = -1;
	cin >> n >> m;
	while (m--) cin >> x >> y, lj[x].push_back(y), lj[y].push_back(x);
	while (la != ans)
	{
		memset(ed + 1, 0, n << 2); la = ans;
		for (i = 1; i <= n; i++) if (!lk[i]) ans += dfs(cnt = i);
	}
	cout << ans << '\n';
	for (i = 1; i <= n; i++) cout << lk[i] << " \n"[i == n];
}

