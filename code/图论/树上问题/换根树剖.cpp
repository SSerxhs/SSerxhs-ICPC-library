
int find(int x,int y)//找到 y 向 x 的子树
{
	while ((top[x]!=top[y])&&(f[top[x]]!=y)) x=f[top[x]];
	if (top[x]==top[y]) return hc[y];
	return top[x];
}
