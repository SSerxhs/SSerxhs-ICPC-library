#include "bits/stdc++.h"
using namespace std;
const int N = 2e5 + 2;
int a[N], z[N], y[N], wz[N], b[N], d[N], bel[N], ans[N], st[N][2], pos[N][2];
void qs(int l, int r)
{
	int i = l, j = r, m = bel[z[l + r >> 1]], mm = y[l + r >> 1];
	while (i <= j)
	{
		while ((bel[z[i]] < m) || (bel[z[i]] == m) && (y[i] < mm)) ++i;
		while ((bel[z[j]] > m) || (bel[z[j]] == m) && (y[j] > mm)) --j;
		if (i <= j)
		{
			swap(wz[i], wz[j]);
			swap(z[i], z[j]);
			swap(y[i++], y[j--]);
		}
	}
	if (i < r) qs(i, r);
	if (l < j) qs(l, j);
}
int main()
{
	ios::sync_with_stdio(false);
	cin.tie(0);
	cin >> n;
	ksiz = sqrt(n);
	for (i = 1; i <= n; i++) { cin >> a[i]; b[i] = a[i]; bel[i] = (i - 1) / ksiz + 1; }
	sort(b + 1, b + n + 1);
	d[gs = 1] = b[1];
	for (i = 2; i <= n; i++) if (b[i] != b[i - 1]) d[++gs] = b[i];
	for (i = 1; i <= n; i++) a[i] = lower_bound(d + 1, d + gs + 1, a[i]) - d;
	cin >> m; assert(int(n / sqrt(m)));
	for (i = 1; i <= m; i++)  cin >> z[i] >> y[wz[i] = i];
	qs(1, m);
	for (i = 1; i <= m; i++)
	{
		if (bel[z[i]] > bel[z[i - 1]])
		{
			while (l <= r) { pos[a[l]][0] = pos[a[l]][1] = 0; ++l; }na = 0;
			if (bel[z[i]] == bel[y[i]])
			{
				for (j = z[i]; j <= y[i]; j++) if (pos[a[j]][0]) na = max(na, j - pos[a[j]][0]); else pos[a[j]][0] = j;
				ans[wz[i]] = na; for (j = z[i]; j <= y[i]; j++) pos[a[j]][0] = 0; na = 0; l = ksiz * bel[z[i]]; r = l - 1;
				continue;
			}
			l = ksiz * bel[z[i]]; r = l - 1; na = 0;
		}
		if (bel[z[i]] == bel[y[i]])
		{
			while (l <= r) { pos[a[l]][0] = pos[a[l]][1] = 0; ++l; }na = 0;
			for (j = z[i]; j <= y[i]; j++) if (pos[a[j]][0]) na = max(na, j - pos[a[j]][0]); else pos[a[j]][0] = j;
			ans[wz[i]] = na; for (j = z[i]; j <= y[i]; j++) pos[a[j]][0] = 0;
			l = ksiz * bel[z[i]]; r = l - 1; na = 0;
			continue;
		}
		while (r < y[i])
		{
			x = a[++r]; pos[x][1] = r;
			if (!pos[x][0]) pos[x][0] = r; else na = max(na, r - pos[x][0]);
		}c = na;
		while (l > z[i])
		{
			x = a[--l]; st[++tp][0] = x; st[tp][1] = pos[x][0];
			pos[x][0] = l;
			if (!pos[x][1])
			{
				st[++tp][0] = x + n; st[tp][1] = 0;
				pos[x][1] = l;
			}
			else na = max(na, pos[x][1] - l);
		}
		ans[wz[i]] = na; na = c; ++tp; l = ksiz * bel[z[i]];
		while (--tp) if (st[tp][0] <= n) pos[st[tp][0]][0] = st[tp][1]; else pos[st[tp][0] - n][1] = st[tp][1];
	}
	for (i = 1; i <= m; i++) cout << ans[i] << "\n";
}

