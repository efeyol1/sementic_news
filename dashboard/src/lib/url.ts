/**
 * Dashboard URL builders that preserve `?country` (and `?date`) across
 * navigation. Sprint 7.5 — central helper introduced after every link
 * builder in the dashboard was dropping the country param when switching
 * dates or drilling into clusters, silently routing the user back to the
 * Turkey default.
 *
 * Use these instead of constructing `?date=...` strings inline. The
 * preserveQueryParams() form is for in-place navigations (e.g. a date
 * picker emitting `router.push()`); the build* helpers are for `<Link
 * href=...>` and any other declarative href construction.
 */

import type { ReadonlyURLSearchParams } from "next/navigation";

type ParamSource = ReadonlyURLSearchParams | URLSearchParams | string;

/**
 * Apply `overrides` on top of the existing `searchParams`, dropping any
 * key whose value is `undefined` or an empty string. Returns
 * `${pathname}?${qs}` or just `pathname` when no params remain.
 */
export function preserveQueryParams(
  pathname: string,
  searchParams: ParamSource,
  overrides: Record<string, string | undefined> = {},
): string {
  const params =
    typeof searchParams === "string"
      ? new URLSearchParams(searchParams)
      : new URLSearchParams(searchParams.toString());
  for (const [key, value] of Object.entries(overrides)) {
    if (value === undefined || value === "") {
      params.delete(key);
    } else {
      params.set(key, value);
    }
  }
  const qs = params.toString();
  return qs ? `${pathname}?${qs}` : pathname;
}

/** `/?date=...&country=...` (empty values are omitted). */
export function buildDashboardUrl(date?: string, country?: string): string {
  return appendParams("/", { date, country });
}

/** `/topic/${id}?date=...&country=...`. */
export function buildTopicUrl(
  clusterId: number,
  date?: string,
  country?: string,
): string {
  return appendParams(`/topic/${clusterId}`, { date, country });
}

/** `/sources?date=...&country=...`. */
export function buildSourcesUrl(date?: string, country?: string): string {
  return appendParams("/sources", { date, country });
}

function appendParams(
  pathname: string,
  fields: Record<string, string | undefined>,
): string {
  const params = new URLSearchParams();
  for (const [key, value] of Object.entries(fields)) {
    if (value) params.set(key, value);
  }
  const qs = params.toString();
  return qs ? `${pathname}?${qs}` : pathname;
}
