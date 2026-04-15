#include "bits/stdc++.h"
using namespace std;
typedef long long ll;
const int N = 3.2e5 + 2;
ll s[N];
int ss[N], ys[N], gs = 0;
bool ed[N];
ll cal(ll m)
{
	static ll g[N << 1], fs[N << 1];
	ll i, j, k, x;
	int n;
	int p, q, cnt;
	n = round(sqrt(m));
	q = lower_bound(ss + 1, ss + gs + 1, n) - ss;
	memset(g, 0, sizeof(g)); memset(ys, 0, sizeof(ys)); cnt = n - 1;
	for (i = n; i <= m; i = j + 1) { j = m / (m / i); ++cnt; }int ct = cnt++;
	for (i = 1; i <= m; i = j + 1)
	{
		j = m / (k = m / i);
		if (k <= n) g[fs[k] = k] = k - 1; else { g[ys[j] = --cnt] = k - 1; fs[cnt] = k; }
	}cnt = ct;
	for (j = 1; j <= q; j++) for (i = cnt; (ll)ss[j] * ss[j] <= fs[i]; i--)
	{
		x = fs[i] / ss[j]; if (x > n) x = ys[m / x];
		g[i] -= g[x] - j + 1;
	}
	return g[cnt];//这里 g[cnt-i+1] 表示的是 [1,m/i] 的答案
}
int main()
{
	int n, i, j, t;
	n = 3.2e5;
	for (i = 2; i <= n; i++)
	{
		if (!ed[i]) ss[++gs] = i;
		for (j = 1; (j <= gs) && (i * ss[j] <= n); j++)
		{
			ed[i * ss[j]] = 1;
			if (i % ss[j] == 0) break;
		}
	}
	s[1] = ss[1];
	for (i = 2; i <= gs; i++) s[i] = s[i - 1] + ss[i];
	t = 1;
	ll m;
	while (t--) cin >> m, cout << cal(m) << '\n';
}
