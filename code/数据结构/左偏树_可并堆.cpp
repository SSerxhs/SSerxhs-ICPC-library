struct left_tree//小根堆，大根堆需要改的地方注释了
{
	int jl[N], v[N], f[N], c[N][2], tf[N], n;//tf只有删非堆顶才用
	bool ed[N];
	void init(const int nn, const int *a)
	{
		jl[0] = -1; n = nn;
		memset(jl + 1, 0, n << 2);
		memset(tf + 1, 0, n << 2);//同上
		memset(c + 1, 0, n << 3);
		memset(ed + 1, 0, n);
		for (int i = 1; i <= n; i++) v[f[i] = i] = a[i];
	}
	int mg(int x, int y)
	{
		if (!(x && y)) return x | y;
		if (v[x] > v[y] || v[x] == v[y] && x > y) swap(x, y);//改
		tf[c[x][1] = mg(c[x][1], y)] = x;//同上
		if (jl[c[x][0]] < jl[c[x][1]]) swap(c[x][0], c[x][1]);
		jl[x] = jl[c[x][1]] + 1;
		return x;
	}
	int getf(int x)
	{
		if (f[x] == x) return x;
		return f[x] = getf(f[x]);
	}
	int merge(int x, int y)
	{
		if (ed[x] || ed[y] || (x = getf(x)) == (y = getf(y))) return x;
		int z = mg(x, y); return f[x] = f[y] = z;
	}
	int getv(int x)//需要自行判断是否存在
	{
		return v[getf(x)];
	}
	int del(int x)//删除堆内最值
	{
		tf[c[x][0]] = tf[c[x][1]] = 0;
		f[c[x][0]] = f[c[x][1]] = f[x] = mg(c[x][0], c[x][1]);
		ed[x] = 1; c[x][0] = c[x][1] = tf[x] = 0; return f[x];
	}
	int del_all(int x)//删除堆内非最值（没验证过）
	{
		int fa = tf[x];
		if (f[c[x][0]] == x) f[c[x][0]] = getf(tf[x]);
		if (f[c[x][1]] == x) f[c[x][1]] = f[tf[x]];
		tf[x] = tf[c[x][0]] = tf[c[x][1]] = 0;
		tf[c[fa][c[fa][1] == x] = mg(c[x][0], c[x][1])] = fa;
		c[x][0] = c[x][1] = 0;
		while (jl[c[fa][0]] < jl[c[fa][1]])
		{
			swap(c[fa][0], c[fa][1]);
			jl[fa] = jl[c[fa][1]] + 1;
			fa = tf[fa];
		}
	}
	void out(int n)
	{
		for (int i = 1; i <= n; i++) printf("%d: c%d&%d f%d v%d\n", i, c[i][0], c[i][1], f[i], v[i]);
	}
};
