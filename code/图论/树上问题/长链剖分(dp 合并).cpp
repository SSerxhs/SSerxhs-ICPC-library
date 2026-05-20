void dfs1(int x)
{
	top[x]=1;
	for (int i=fir[x];i;i=nxt[i]) if (!top[lj[i]])
	{
		dfs1(lj[i]);
		if (len[lj[i]]>len[hc[x]]) hc[x]=lj[i];
	}
	len[x]=len[hc[x]]+1;top[hc[x]]=0;
}
void dfs2(int x)
{
	*f[x]=1;gs[x]=1;
	if (!hc[x]) return;
	ed[x]=1;f[hc[x]]=f[x]+1;
	for (int i=fir[x];i;i=nxt[i]) if (!ed[lj[i]]) dfs2(lj[i]);
	ans[x]=ans[hc[x]]+1;gs[x]=gs[hc[x]];
	if (gs[x]==1) ans[x]=0;
	for (int i=fir[x];i;i=nxt[i]) if ((!ed[lj[i]])&&(lj[i]!=hc[x]))
	{
		int v=lj[i],*p;
		for (int j=0;j<len[v];j++)
		{
			*(p=f[x]+j+1)+=*(f[v]+j);
			if (j+1==ans[x]) {gs[x]=*p;continue;}
			if ((*p>gs[x])||(*p==gs[x])&&(j+1<ans[x])) {gs[x]=*p;ans[x]=j+1;}
		}
	}
	gs[x]=*(f[x]+ans[x]);
	ed[x]=0;
}
