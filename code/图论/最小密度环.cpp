#include "bits/stdc++.h"
using namespace std;
const int N = 3e3 + 5, M = 1e4 + 5;
const double inf = 1e18;
int u[M], v[M];
double f[N][N], w[M];
int main()
{
	ios::sync_with_stdio(0); cin.tie(0);
	cout << setiosflags(ios::fixed) << setprecision(8);
	int n, m, i, j;
	cin >> n >> m;
	for (i = 1; i <= m; i++)
		cin >> u[i] >> v[i] >> w[i];
	++n;
	for (i = 1; i <= n; i++)
	{
		fill_n(f[i] + 1, n, inf);
		for (j = 1; j <= m; j++)
			f[i][v[j]] = min(f[i][v[j]], f[i - 1][u[j]] + w[j]);
	}
	double ans = inf;
	for (i = 1; i < n; i++) if (f[n][i] != inf)
	{
		double r = -inf;
		for (j = 1; j < n; j++) r = max(r, (f[n][i] - f[j][i]) / (n - j));
		ans = min(ans, r);
	}
	cout << ans << endl;
}
