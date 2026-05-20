int lca(int x, int y)
{
	int i;
	if (dep[x] < dep[y]) swap(x, y);
	for (i = lm[x]; dep[x] != dep[y]; i--) if (dep[f[x][i]] >= dep[y]) x = f[x][i];
	if (x == y) return x;
	for (i = lm[x]; f[x][0] != f[y][0]; i--) if (f[x][i] != f[y][i])
	{
		x = f[x][i]; y = f[y][i];
	}
	return f[x][0];
}
void dfs(int x)
{
	s[x] = 1;
	int i;
	for (i = sfir[x]; i; i = snxt[i])
	{
		dfs(slj[i]);
		s[x] += s[slj[i]];
	}
}
int main()
{
	dep[0] = -1;
	cin >> n;
	for (i = 1; i <= n; i++)
	{
		cin >> x;
		while (x)
		{
			add(x, i);
			cin >> x;
		}
	}
	dl[tou = wei = 1] = ++n;
	for (i = 1; i < n; i++) if (!rd[i]) add(n, i);
	while (tou <= wei)
	{
		for (i = fir[x = dl[tou++]]; i; i = nxt[i]) if (--rd[lj[i]] == 0) dl[++wei] = lj[i];
		if (i = ffir[x])
		{
			y = flj[i];
			while (i = fnxt[i]) y = lca(y, flj[i]);
			f[x][0] = y;
		}
		else y = 0;
		sadd(y, x);
		f[x][0] = y;
		for (i = 1; i <= 16; i++) if (0 == (f[x][i] = f[f[x][i - 1]][i - 1]))
		{
			lm[x] = i;
			break;
		}
		dep[x] = dep[y] + 1;
	}
	dfs(n);
	for (i = 1; i < n; i++) printf("%d\n", s[i] - 1);
}
