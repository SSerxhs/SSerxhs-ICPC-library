int a[M][N], b[M][N << 1], c[M][N << 1][2], len[M], ans[M];
int n, m, qs, p, q, d, i, j, x, y, la;
int main()
{
	cin >> n >> m >> qs >> d;
	for (j = 1; j <= m; j++) for (i = 0; i < n; i++) cin >> a[j][i];
	for (j = 1; j <= m; j++) a[j][n] = inf + j; ++n;
	for (i = 0; i < n; i++) b[m][i] = a[m][i], c[m][i][0] = i;
	len[m] = n;
	for (j = m - 1; j; j--)
	{
		p = 0, q = 1;
		while (p < n && q < len[j + 1])
			if (a[j][p] < b[j + 1][q]) b[j][len[j]] = a[j][p], c[j][len[j]][0] = p++, c[j][len[j]++][1] = q;
			else b[j][len[j]] = b[j + 1][q], c[j][len[j]][0] = p, c[j][len[j]++][1] = q, q += 2;
		while (p < n) b[j][len[j]] = a[j][p], c[j][len[j]][0] = p++, c[j][len[j]++][1] = q;
		while (q < len[j + 1]) b[j][len[j]] = b[j + 1][q], c[j][len[j]][0] = p, c[j][len[j]++][1] = q, q += 2;
	}
	for (int ii = 1; ii <= qs; ii++)
	{
		cin >> x; x ^= la;
		y = lower_bound(b[1], b[1] + len[1], x) - b[1];
		ans[1] = a[1][c[1][y][0]]; y = c[1][y][1];//下标是c[1][y][0]
		for (j = 2; j <= m; j++)
		{
			if (y && b[j][y - 1] >= x) --y;
			ans[j] = a[j][c[j][y][0]];//下标是c[j][y][0]
			y = c[j][y][1];
		}
		la = 0;
		for (i = 1; i <= m; i++) la ^= ans[i] > inf ? 0 : ans[i];
		if (ii % d == 0) cout << la << '\n';
	}
}

