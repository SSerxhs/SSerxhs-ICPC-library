int f[N][N],jl[N][N];
int n,m,c,ans=inf,i,j,k,x,y,z;
int main()
{
	cin>>n>>m;
	memset(f,0x3f,sizeof(f));
	memset(jl,0x3f,sizeof(jl));
	while (m--)
	{
		cin>>x>>y>>z;
		jl[x][y]=jl[y][x]=f[x][y]=f[y][x]=min(f[y][x],z);
	}
	for (k=1;k<=n;k++)
	{
		for (i=1;i<k;i++) if (jl[k][i]!=jl[0][0]) for (j=1;j<i;j++)
			if (jl[k][j]!=jl[0][0]) ans=min(ans,jl[k][i]+jl[k][j]+f[i][j]);
		for (i=1;i<=n;i++) if (i!=k) for (j=1;j<=n;j++)
			if ((j!=i)&&(j!=k)) f[i][j]=min(f[i][j],f[i][k]+f[k][j]);
	}
	if (ans==inf) cout<<"No solution.\n"; else cout<<ans<<endl;
}
