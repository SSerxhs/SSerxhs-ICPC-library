void dfs1(int x)
{
	int i;
	for (i = 1; i <= er[dep[x] - 1]; i++) f[x][i] = f[f[x][i - 1]][i - 1]; md[x] = dep[x];
	for (i = fir[x]; i; i = nxt[i]) { dep[lj[i]] = dep[x] + 1; dfs1(lj[i]); if (md[lj[i]] > md[dc[x]]) dc[x] = lj[i]; }
	if (dc[x]) md[x] = md[dc[x]];
}
void dfs2(int x)
{
	int i;
	if (dc[x])
	{
		top[dc[x]] = top[x];
		dfs2(dc[x]);
		for (i = fir[x]; i; i = nxt[i]) if (lj[i] != dc[x]) dfs2(top[lj[i]] = lj[i]);
	}
	if (x == top[x])
	{
		c = md[x] - dep[x]; y = x; up[x].push_back(x); down[x].push_back(x);
		for (i = 1; (i <= c) && (y = f[y][0]); i++) up[x].push_back(y); y = x;
		for (i = 1; i <= c; i++) down[x].push_back(y = dc[y]);
	}
}
int main()
{
	int n, q, ans = 0, x, y, c, i;
	ll ta = 0;
	cin >> n >> q >> s;
	for (i = 1; i <= n; i++) { cin >> f[i][0]; if (f[i][0] == 0) rt = i; else add(f[i][0], i); }
	for (i = 2; i <= n; i++) er[i] = er[i >> 1] + 1; dep[rt] = 1;
	dfs1(rt); dfs2(top[rt] = rt);
	for (i = 1; i <= q; i++)
	{
		x = (get(s) ^ ans) % n + 1; y = (get(s) ^ ans) % dep[x];
		//此时计算 x 的 y 级祖先。结果在 ans 中。
		if (y == 0) { ans = x; ta ^= (ll)i * ans; continue; }
		c = dep[x] - y; x = top[f[x][er[y]]];
		if (dep[x] > c) ans = up[x][dep[x] - c]; else ans = down[x][c - dep[x]];
		ta ^= (ll)i * ans;
	}
	cout << ta << endl;
}
