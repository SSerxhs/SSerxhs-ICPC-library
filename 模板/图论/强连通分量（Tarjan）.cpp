int dfn[N],low[N],st[N],f[N],fs,tp,id;
bool ed[N];
void tarjan(int u)
{
	dfn[u]=low[u]=++id;
	ed[u]=1;st[++tp]=u;
	for (int v:e[u]) if (dfn[v])
	{
		if (ed[v]) low[u]=min(low[u],dfn[v]);
	} else tarjan(v),low[u]=min(low[u],low[v]);
	if (dfn[u]==low[u])
	{
		++fs;
		do
		{
			f[st[tp]]=fs;
			ed[st[tp]]=0;
		} while (st[tp--]!=u);
	}
}
