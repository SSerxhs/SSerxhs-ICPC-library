#include "bits/stdc++.h"
using namespace std;
typedef long long ll;
const int N = 202, M = 10002;
struct P
{
	int x, y;
	P(int a = 0, int b = 0) :x(a), y(b) { }
	bool operator<(const P &o) const { return (ll)x * y < (ll)o.x * o.y || (ll)x * y == (ll)o.x * o.y && x < o.x; }
};
struct Q
{
	int u, v, x, y, val;
	bool operator<(const Q &o) const { return val < o.val; }
};
P ans = P(1e9, 1e9), l, r;
Q a[M];
int f[N];
int n, m, i;
int getf(int x)
{
	if (f[x] == x) return x;
	return f[x] = getf(f[x]);
}
P sol1()
{
	P r = P(0, 0);
	for (i = 1; i <= n; i++) f[i] = i;
	sort(a + 1, a + m + 1);
	for (i = 1; i <= m; i++) if (getf(a[i].u) != getf(a[i].v))
	{
		f[f[a[i].u]] = f[a[i].v];
		r.x += a[i].x, r.y += a[i].y;
	}
	return r;
}
void sol2(P l, P r)
{
	for (i = 1; i <= m; i++) a[i].val = (r.x - l.x) * a[i].y + (l.y - r.y) * a[i].x;
	P np = sol1();
	ans = min(ans, np);
	if ((ll)(r.x - l.x) * (np.y - l.y) - (ll)(r.y - l.y) * (np.x - l.x) >= 0) return;
	sol2(l, np); sol2(np, r);
}
int main()
{
	cin >> n >> m;
	for (i = 1; i <= m; i++) cin >> a[i].u >> a[i].v >> a[i].x >> a[i].y, ++a[i].u, ++a[i].v;
	for (i = 1; i <= m; i++) a[i].val = a[i].x; l = sol1();
	for (i = 1; i <= m; i++) a[i].val = a[i].y; r = sol1();
	ans = min(ans, min(l, r)); sol2(l, r);
	cout << ans.x << ' ' << ans.y << endl;
}
