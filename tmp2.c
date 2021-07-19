int main()
{
  int a[MAX];
  int i;
  int win;
  int n;
  int t;
  scanf("%d", &n);
  for (i = 0; i < n; i++)
  {
    scanf("%d ", &a[i]);
  }

  for (i = 0, win = n - 1; i < win; i++, win--)
  {
    t = a[i];
    a[i] = a[win];
    a[win] = t;
  }

  for (i = 0; i < n; i++)
  {
    if (i == (n - 1))
    {
      printf("%d", a[i]);
    }
    else
    {
      printf("%d ", a[i]);
    }
  }

  return 0;
}