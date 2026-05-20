int c[N][2], p[N], st[N];
int main()
{
	int i, n, tp = 0;
	cin >> n;
	for (i = 1; i <= n; i++)
	{
		cin >> p[i]; st[tp + 1] = 0;
		while ((tp) && (p[st[tp]] > p[i])) --tp;
		c[c[st[tp]][1] = i][0] = st[tp + 1]; st[++tp] = i;
	}
}

