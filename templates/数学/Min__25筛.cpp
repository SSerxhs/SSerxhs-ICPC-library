const int N = 1e5 + 2, p = 1e9 + 7, i6 = 166666668;
ll fs[N << 1], m;
int ss[N], ys[N << 1], s[N], f[N << 1], g[N << 1], ls[N << 1], cs[N << 1];
int gs, n, i, j, k, cnt, ct, ans, sq;
bool ed[N];
int S(ll n, int x)
{
	int r, i, j, l;
	ll k;
	if (ss[x] >= n) return 0;
	if (n > sq) r = g[ys[m / n]]; else r = g[n];
	if ((r = r - s[x]) < 0) r += p;
	for (i = x + 1; (ll)ss[i] * ss[i] <= n; i++) for (j = 1, k = ss[i]; k <= n; j++, k *= ss[i])
	{
		l = (k - 1) % p;
		r = (r + (ll)l * (l + 1) % p * ((j != 1) + S(n / k, i))) % p;
	}
	return r;
}
int main()
{
	n = 1e5;
	for (i = 2; i <= n; i++)
	{
		if (!ed[i]) ss[++gs] = i;
		for (j = 1; (j <= gs) && (i * ss[j] <= n); j++)
		{
			ed[i * ss[j]] = 1;
			if (i % ss[j] == 0) break;
		}
	}ss[gs + 1] = 1e6;
	s[1] = ss[1] * ss[1];
	for (i = 2; i <= gs; i++) s[i] = (s[i - 1] + (ll)ss[i] * ss[i]) % p;//s 是多项式在素数位置的前缀和
	memcpy(cs, s, sizeof(s));
	ll i, j, k, x, z; scanf("%lld", &m);
	sq = n = sqrt(m); while ((ll)(n + 1) * (n + 1) <= m) ++n;
	cnt = n - 1;
	for (i = n; i <= m; i = j + 1) { j = m / (m / i); ++cnt; }ct = cnt++;
	for (i = 1; i <= m; i = j + 1)
	{
		j = m / (k = m / i);
		if (k <= n) g[fs[k] = k] = (k * (k + 1) * (k << 1 | 1) / 6 - 1) % p;//这里是多项式前缀和（不含1）
		else
		{
			z = k % p;//一样
			g[ys[j] = --cnt] = (z * (z + 1) % p * (z << 1 | 1) % p + p - 6) * i6 % p; fs[cnt] = k;
		}
	}
	cnt = ct;
	for (j = 1; (j <= gs) && (z = (ll)ss[j] * ss[j]); j++) for (i = cnt; z <= fs[i]; i--)
	{
		x = fs[i] / ss[j]; if (x > n) x = ys[m / x];
		g[i] = (g[i] + (ll)(p - ss[j]) * ss[j] % p * (g[x] - s[j - 1] + p)) % p;//另一处需要修改的
	}
	memcpy(ls, g, sizeof(g));
	s[1] = ss[1];
	for (i = 2; i <= gs; i++) s[i] = s[i - 1] + ss[i];
	cnt = n - 1;
	for (i = n; i <= m; i = j + 1) { j = m / (m / i); ++cnt; }ct = cnt++;
	for (i = 1; i <= m; i = j + 1)
	{
		j = m / (k = m / i);
		if (k <= n) g[fs[k] = k] = ((k * (k + 1) >> 1) - 1) % p;
		else
		{
			z = k % p;
			g[ys[j] = --cnt] = (z * (z + 1) - 2 >> 1) % p; fs[cnt] = k;
		}
	}
	cnt = ct;
	for (j = 1; (j <= gs) && (z = (ll)ss[j] * ss[j]); j++) for (i = cnt; z <= fs[i]; i--)
	{
		x = fs[i] / ss[j]; if (x > n) x = ys[m / x];
		g[i] = (g[i] + (ll)(p - ss[j]) * (g[x] - s[j - 1] + p)) % p;
	}
	for (i = 1; i <= cnt; i++) if ((g[i] = ls[i] - g[i]) < 0) g[i] += p;
	for (i = 1; i <= gs; i++) if ((s[i] = cs[i] - s[i]) < 0) s[i] += p;
	ans = S(m, 0) + 1; if (ans == p) ans = 0; printf("%d", ans);
}
