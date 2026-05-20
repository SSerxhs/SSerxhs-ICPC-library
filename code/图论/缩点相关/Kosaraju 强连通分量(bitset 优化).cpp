void dfs1(int x)
{
	int i; ed[x] = 0;
	for (i = (lj[x] & ed)._Find_first(); i <= n; i = (lj[x] & ed)._Find_next(i)) dfs1(i);
	sx[--tp] = x;
}
void dfs2(int x)
{
	int i; ed[x] = 0; tv[f[x] = f[0]] += v[x];
	for (i = (fj[x] & ed)._Find_first(); i <= n; i = (fj[x] & ed)._Find_next(i)) dfs2(i);
}
int main()
{
	cin >> n >> m;
	tp = n + 1;
	for (i = 1; i <= n; i++) { ed[i] = 1; cin >> v[i]; }
	for (i = 1; i <= m; i++)
	{
		cin >> x >> y;
		lj[x][y] = 1; fj[y][x] = 1; lb[i][0] = x; lb[i][1] = y;
	}
	for (i = 1; i <= n; i++) if (ed[i]) dfs1(i);
	ed.set();
	for (i = 1; i <= n; i++) if (ed[sx[i]]) { ++f[0]; dfs2(sx[i]); }
	for (i = 1; i <= m; i++) if (f[lb[i][0]] != f[lb[i][1]])
	{
		flj[f[lb[i][0]]].push_back(f[lb[i][1]]); ++rd[f[lb[i][1]]];
	}
	for (i = 1; i <= f[0]; i++) if (!rd[i]) dl[++wei] = i;
	while (tou <= wei)
	{
		x = dl[tou++]; g[x] += tv[x];
		for (i = 0; i < flj[x].size(); i++)
		{
			g[flj[x][i]] = max(g[flj[x][i]], g[x]);
			if (--rd[flj[x][i]] == 0) dl[++wei] = flj[x][i];
		}
	}
	for (i = 1; i <= f[0]; i++) ans = max(ans, g[i]); printf("%d", ans);
}
