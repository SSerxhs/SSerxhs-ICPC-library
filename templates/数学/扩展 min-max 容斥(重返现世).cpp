scanf("%d%d%d", &n, &q, &m); inv[1] = 1; q = n + 1 - q;
for (i = 2; i <= m; i++) inv[i] = p - (ll)p / i * inv[p % i] % p;
for (i = 1; i <= n; i++) scanf("%d", a + i); f[0][0] = 1;
for (j = 1; j <= n; j++) for (i = q; i; i--) for (k = m; k >= a[j]; k--) if ((f[i][k] = f[i][k] + f[i - 1][k - a[j]] - f[i][k - a[j]]) >= p) f[i][k] -= p; else if (f[i][k] < 0) f[i][k] += p;
for (i = 1; i <= m; i++) ans = (ans + (ll)f[q][i] * inv[i]) % p;
ans = (ll)ans * m % p; printf("%d", ans);

