void dfs1(int x)
{
	siz[x] = zdep[x] = 1;
	int i;
	for (i = fir[x]; i; i = nxt[i]) if (lj[i] != f[x])
	{
		dep[lj[i]] = dep[f[lj[i]] = x] + 1;
		dfs1(lj[i]);
		siz[x] += siz[lj[i]];
		if (siz[hc[x]] < siz[lj[i]]) hc[x] = lj[i];
		zdep[x] = max(zdep[x], zdep[lj[i]] + 1);
	}
}
void cal(int x)
{
	int i;
	dl[tou = wei = 1] = x;
	while (tou <= wei)
	{
		++dp[dep[x = dl[tou++]]];
		if ((dp[dep[x]] > dp[zd]) || (dp[dep[x]] == dp[zd]) && (dep[x] < zd)) zd = dep[x];
		for (i = fir[x]; i; i = nxt[i]) if (lj[i] != f[x]) dl[++wei] = lj[i];
	}
}
void dfs2(int x)
{
	if (!hc[x])
	{
		if (++dp[dep[x]] > dp[zd]) zd = dep[x];
		return;
	}
	int i;
	for (i = fir[x]; i; i = nxt[i]) if ((lj[i] != f[x]) && (lj[i] != hc[x]))
	{
		dfs2(lj[i]);
		memset(dp + dep[lj[i]], 0, zdep[lj[i]] << 2);
	}
	dfs2(hc[x]);
	dp[dep[x]] = 1;
	if (dp[zd] <= 1) zd = dep[x];
	for (i = fir[x]; i; i = nxt[i]) if ((lj[i] != f[x]) && (lj[i] != hc[x])) cal(lj[i]);
	ans[x] = zd - dep[x];
}
