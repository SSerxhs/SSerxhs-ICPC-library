const int N = 1.1e6 + 2;
int c[N][2], v[N], w[N], s[N];
int n, i, x, y, ds, val, kth, p, q, z, rt, la, m, ans;
void pushup(const int x)
{
	s[x] = s[c[x][0]] + s[c[x][1]] + 1;
}
void split_val(int now, int &x, int &y)//调用外部val,相等归入y
{
	if (!now) return x = y = 0, void();
	if (val <= v[now]) split_val(c[y = now][0], x, c[now][0]);
	else split_val(c[x = now][1], c[now][1], y);
	pushup(now);
}
void split_kth(int now, int &x, int &y)//调用外部kth，左子树大小为 kth
{
	if (!now) return x = y = 0, void();
	if (kth <= s[c[now][0]]) split_kth(c[y = now][0], x, c[now][0]);
	else kth -= s[c[now][0]] + 1, split_kth(c[x = now][1], c[now][1], y);
	pushup(now);
}
int merge(int x, int y)//小根ver.
{
	if (!(x && y)) return x | y;
	if (w[x] < w[y]) { c[x][1] = merge(c[x][1], y); pushup(x); return x; }
	else { c[y][0] = merge(x, c[y][0]); pushup(y); return y; }
}
int main()
{
	cin >> n >> m; srand(998244353);
	for (i = 1; i <= n; i++)
	{
		cin >> x;
		val = v[++ds] = x;
		w[ds] = rand();
		s[ds] = 1;
		split_val(rt, p, q);
		rt = merge(merge(p, ds), q);
	}
	while (m--)
	{
		cin >> y >> x;
		x ^= la;
		if (y == 4)//找到第 x 小的
		{
			kth = x; split_kth(rt, p, q); x = p;
			while (c[x][1]) x = c[x][1];
			ans ^= (la = v[x]); rt = merge(p, q);
			continue;
		}
		val = x;//注意这一步
		if (y == 1)//插入 x
		{
			v[++ds] = x; w[ds] = rand(); s[ds] = 1;
			split_val(rt, p, q); rt = merge(merge(p, ds), q);
			continue;
		}
		if (y == 2)//删除一个 x
		{
			split_val(rt, p, q);
			z = q;
			if (q)
			{
				i = q;
				while (c[i][0]) i = c[i][0];
				if (v[i] == x)
				{
					kth = 1;
					split_kth(q, i, z);
				}
			}
			rt = merge(p, z); continue;
		}
		if (y == 3)//询问 x 的排名（比 x 小的数字个数 +1）
		{
			split_val(rt, p, q); ans ^= (la = s[p] + 1);
			rt = merge(p, q); continue;
		}
		if (y == 5)//询问比 x 小的最大值
		{
			split_val(rt, p, q); x = p;
			while (c[x][1]) x = c[x][1]; ans ^= (la = v[x]);
			rt = merge(p, q); continue;
		}
		++val; split_val(rt, p, q); x = q;//询问比 x 大的最小值
		while (c[x][0]) x = c[x][0];
		ans ^= (la = v[x]); rt = merge(p, q);
	}
	cout << ans << endl;
}

