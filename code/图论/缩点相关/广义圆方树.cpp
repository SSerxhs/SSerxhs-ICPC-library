void dfs(int u)
{
	dfn[u] = low[u] = ++id;
	st[++tp] = u;
	for (int v : e[u]) if (dfn[v]) low[u] = min(low[u], dfn[v]); else
	{
		dfs(v);
		low[u] = min(low[u], low[v]);
		if (dfn[u] <= low[v])
		{
			vector cur = {u};
			do
			{
				cur.push_back(st[tp]);
			} while (st[tp--] != v);
			ans.push_back(cur);
		}
	}
}

