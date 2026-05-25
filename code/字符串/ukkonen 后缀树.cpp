void dfs(int x, int lf)
{
	if (!fir[x])
	{
		siz[x][1] = 1;
		return;
	}
	int i, j;
	for (i = fir[x]; i; i = nxt[i])
	{
		j = c[x][lj[i]];
		if ((f[j] <= m) && (t[j] >= m)) ++siz[x][0];
		dfs(zd[j], t[j] - f[j] + 1);
		siz[x][0] += siz[zd[j]][0];
		siz[x][1] += siz[zd[j]][1];
		if ((t[j] == n) && (f[j] <= m)) --siz[x][1];
	}
	ans += (ll)siz[x][0] * siz[x][1] * lf;
}
void add(int a, int b, int cc, int d)
{
	zd[++bbs] = b;
	t[bbs] = d;
	c[a][s[f[bbs] = cc]] = bbs;
}
void add(int x, int y)
{
	lj[++bs] = y;
	nxt[bs] = fir[x];
	fir[x] = bs;
}
s[++m] = 26;
fa[1] = point = ds = 1;
for (i = 1; i <= m; i++)
{
	ad = 0; ++remain;
	while (remain)
	{
		if (r == 0) edge = i;
		if ((j = c[point][s[edge]]) == 0)
		{
			fa[++ds] = 1;
			fa[ad] = point;
			add(ad = point, ds, edge, m);
			add(point, s[edge]);
		}
		else
		{
			if ((t[j] != m) && (t[j] - f[j] + 1 <= r))
			{
				r -= t[j] - f[j] + 1;
				edge += t[j] - f[j] + 1;
				point = zd[j];
				continue;
			}
			if (s[f[j] + r] == s[i]) { ++r; fa[ad] = point; break; }
			fa[fa[ad] = ++ds] = 1;
			add(ad = ds, zd[j], f[j] + r, t[j]);
			add(ds, s[i]); add(ds, s[f[j] + r]); fa[++ds] = 1;
			add(ds - 1, ds, i, m);
			zd[j] = ds - 1; t[j] = f[j] + r - 1;
		}
		--remain;
		if ((r) && (point == 1))
		{
			--r; edge = i - remain + 1;
		}
		else point = fa[point];
	}
}
for (i = 1; i <= ds; i++) for (j = fir[i]; j; j = nxt[j]) { len[j] = t[c[i][lj[j]]] - f[c[i][lj[j]]] + 1; lj[j] = zd[c[i][lj[j]]]; }
