/**
 * Timezone-Aware Tests
 *
 * Tests for date/time handling to ensure correct behavior
 * across different timezones and edge cases.
 */

import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';

// ============================================================================
// Timezone Utilities
// ============================================================================

/**
 * Format a date in a specific timezone
 */
function formatInTimezone(date: Date, timezone: string, options?: Intl.DateTimeFormatOptions): string {
  return new Intl.DateTimeFormat('en-US', {
    timeZone: timezone,
    ...options,
  }).format(date);
}

/**
 * Get hour in a specific timezone
 */
function getHourInTimezone(date: Date, timezone: string): number {
  const formatted = new Intl.DateTimeFormat('en-US', {
    timeZone: timezone,
    hour: 'numeric',
    hour12: false,
  }).format(date);
  return parseInt(formatted, 10);
}

/**
 * Get day of week in a specific timezone (0 = Sunday, 6 = Saturday)
 */
function getDayInTimezone(date: Date, timezone: string): number {
  const formatted = new Intl.DateTimeFormat('en-US', {
    timeZone: timezone,
    weekday: 'short',
  }).format(date);
  const days = ['Sun', 'Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat'];
  return days.indexOf(formatted);
}

/**
 * Convert day from Sunday=0 format to Monday=0 format
 */
function adjustDayForMonday(sundayBasedDay: number): number {
  return sundayBasedDay === 0 ? 6 : sundayBasedDay - 1;
}

/**
 * Format relative time (e.g., "5 minutes ago", "in 2 hours")
 */
function formatRelativeTime(date: Date, baseDate: Date = new Date()): string {
  const diffMs = date.getTime() - baseDate.getTime();
  const diffMinutes = Math.round(diffMs / 60000);
  const diffHours = Math.round(diffMs / 3600000);
  const diffDays = Math.round(diffMs / 86400000);

  if (Math.abs(diffMinutes) < 60) {
    if (diffMinutes === 0) return 'just now';
    if (diffMinutes > 0) return `in ${diffMinutes} minute${diffMinutes !== 1 ? 's' : ''}`;
    return `${Math.abs(diffMinutes)} minute${Math.abs(diffMinutes) !== 1 ? 's' : ''} ago`;
  }

  if (Math.abs(diffHours) < 24) {
    if (diffHours > 0) return `in ${diffHours} hour${diffHours !== 1 ? 's' : ''}`;
    return `${Math.abs(diffHours)} hour${Math.abs(diffHours) !== 1 ? 's' : ''} ago`;
  }

  if (diffDays > 0) return `in ${diffDays} day${diffDays !== 1 ? 's' : ''}`;
  return `${Math.abs(diffDays)} day${Math.abs(diffDays) !== 1 ? 's' : ''} ago`;
}

/**
 * Parse ISO date string with timezone awareness
 */
function parseISODate(isoString: string): Date {
  return new Date(isoString);
}

/**
 * Format time for display in 12-hour format
 */
function formatHour12(hour: number): string {
  if (hour === 0) return '12a';
  if (hour === 12) return '12p';
  return hour < 12 ? `${hour}a` : `${hour - 12}p`;
}

/**
 * Check if date falls within business hours (9 AM - 5 PM)
 */
function isBusinessHours(date: Date, timezone: string): boolean {
  const hour = getHourInTimezone(date, timezone);
  const day = getDayInTimezone(date, timezone);
  const isWeekday = day >= 1 && day <= 5;
  const isWorkingHour = hour >= 9 && hour < 17;
  return isWeekday && isWorkingHour;
}

/**
 * Get start of day in a specific timezone
 */
function getStartOfDay(date: Date, timezone: string): Date {
  const parts = new Intl.DateTimeFormat('en-US', {
    timeZone: timezone,
    year: 'numeric',
    month: '2-digit',
    day: '2-digit',
  }).formatToParts(date);

  const year = parts.find(p => p.type === 'year')?.value;
  const month = parts.find(p => p.type === 'month')?.value;
  const day = parts.find(p => p.type === 'day')?.value;

  return new Date(`${year}-${month}-${day}T00:00:00`);
}

// ============================================================================
// Tests
// ============================================================================

describe('Timezone Handling Tests', () => {
  describe('Hour Formatting', () => {
    it('should format hours correctly in 12-hour format', () => {
      expect(formatHour12(0)).toBe('12a');
      expect(formatHour12(1)).toBe('1a');
      expect(formatHour12(11)).toBe('11a');
      expect(formatHour12(12)).toBe('12p');
      expect(formatHour12(13)).toBe('1p');
      expect(formatHour12(23)).toBe('11p');
    });

    it('should handle all 24 hours', () => {
      for (let h = 0; h < 24; h++) {
        const formatted = formatHour12(h);
        expect(formatted).toMatch(/^(1[0-2]|[1-9])[ap]$/);
      }
    });
  });

  describe('Day Adjustment', () => {
    it('should convert Sunday=0 format to Monday=0 format', () => {
      expect(adjustDayForMonday(0)).toBe(6); // Sunday -> 6
      expect(adjustDayForMonday(1)).toBe(0); // Monday -> 0
      expect(adjustDayForMonday(2)).toBe(1); // Tuesday -> 1
      expect(adjustDayForMonday(6)).toBe(5); // Saturday -> 5
    });

    it('should handle all days correctly', () => {
      const expected = [6, 0, 1, 2, 3, 4, 5];
      for (let i = 0; i < 7; i++) {
        expect(adjustDayForMonday(i)).toBe(expected[i]);
      }
    });
  });

  describe('Timezone-Aware Date Formatting', () => {
    it('should format date in different timezones', () => {
      // Use a fixed date to avoid timezone issues
      const date = new Date('2024-06-15T12:00:00Z'); // Noon UTC

      const utc = formatInTimezone(date, 'UTC', { hour: 'numeric', hour12: false });
      const ny = formatInTimezone(date, 'America/New_York', { hour: 'numeric', hour12: false });
      const tokyo = formatInTimezone(date, 'Asia/Tokyo', { hour: 'numeric', hour12: false });

      expect(parseInt(utc, 10)).toBe(12); // Noon UTC
      expect(parseInt(ny, 10)).toBe(8); // 8 AM Eastern (during DST)
      expect(parseInt(tokyo, 10)).toBe(21); // 9 PM Japan
    });

    it('should get correct hour in timezone', () => {
      const date = new Date('2024-01-15T12:00:00Z'); // Noon UTC, no DST

      expect(getHourInTimezone(date, 'UTC')).toBe(12);
      expect(getHourInTimezone(date, 'America/New_York')).toBe(7); // EST = UTC-5
      expect(getHourInTimezone(date, 'America/Los_Angeles')).toBe(4); // PST = UTC-8
      expect(getHourInTimezone(date, 'Europe/London')).toBe(12); // GMT
    });

    it('should get correct day in timezone', () => {
      // Late night in UTC, early morning in Asia
      const date = new Date('2024-06-15T23:00:00Z'); // Saturday 11 PM UTC

      const utcDay = getDayInTimezone(date, 'UTC');
      const tokyoDay = getDayInTimezone(date, 'Asia/Tokyo');

      expect(utcDay).toBe(6); // Saturday
      expect(tokyoDay).toBe(0); // Sunday (it's 8 AM Sunday in Tokyo)
    });
  });

  describe('Relative Time Formatting', () => {
    it('should format "just now" correctly', () => {
      const now = new Date();
      expect(formatRelativeTime(now, now)).toBe('just now');
    });

    it('should format minutes ago correctly', () => {
      const now = new Date();
      const past = new Date(now.getTime() - 5 * 60000);
      expect(formatRelativeTime(past, now)).toBe('5 minutes ago');
    });

    it('should format single minute correctly', () => {
      const now = new Date();
      const past = new Date(now.getTime() - 60000);
      expect(formatRelativeTime(past, now)).toBe('1 minute ago');
    });

    it('should format hours ago correctly', () => {
      const now = new Date();
      const past = new Date(now.getTime() - 3 * 3600000);
      expect(formatRelativeTime(past, now)).toBe('3 hours ago');
    });

    it('should format single hour correctly', () => {
      const now = new Date();
      const past = new Date(now.getTime() - 3600000);
      expect(formatRelativeTime(past, now)).toBe('1 hour ago');
    });

    it('should format days ago correctly', () => {
      const now = new Date();
      const past = new Date(now.getTime() - 2 * 86400000);
      expect(formatRelativeTime(past, now)).toBe('2 days ago');
    });

    it('should format future times correctly', () => {
      const now = new Date();
      const future = new Date(now.getTime() + 30 * 60000);
      expect(formatRelativeTime(future, now)).toBe('in 30 minutes');
    });
  });

  describe('ISO Date Parsing', () => {
    it('should parse ISO date strings correctly', () => {
      const isoString = '2024-06-15T12:30:00Z';
      const date = parseISODate(isoString);

      expect(date.getUTCFullYear()).toBe(2024);
      expect(date.getUTCMonth()).toBe(5); // 0-indexed, June = 5
      expect(date.getUTCDate()).toBe(15);
      expect(date.getUTCHours()).toBe(12);
      expect(date.getUTCMinutes()).toBe(30);
    });

    it('should handle timezone offsets in ISO strings', () => {
      const withOffset = '2024-06-15T12:30:00+05:30';
      const date = parseISODate(withOffset);

      // 12:30 + 5:30 offset = 07:00 UTC
      expect(date.getUTCHours()).toBe(7);
      expect(date.getUTCMinutes()).toBe(0);
    });

    it('should handle date-only ISO strings', () => {
      const dateOnly = '2024-06-15';
      const date = parseISODate(dateOnly);

      expect(date.getUTCFullYear()).toBe(2024);
      expect(date.getUTCMonth()).toBe(5);
      expect(date.getUTCDate()).toBe(15);
    });
  });

  describe('Business Hours Check', () => {
    it('should identify business hours correctly', () => {
      // Tuesday 2 PM UTC
      const businessTime = new Date('2024-06-18T14:00:00Z');
      expect(isBusinessHours(businessTime, 'UTC')).toBe(true);
    });

    it('should identify non-business hours correctly', () => {
      // Saturday 2 PM UTC
      const weekend = new Date('2024-06-15T14:00:00Z');
      expect(isBusinessHours(weekend, 'UTC')).toBe(false);
    });

    it('should handle early morning correctly', () => {
      // Tuesday 6 AM UTC (before 9 AM)
      const earlyMorning = new Date('2024-06-18T06:00:00Z');
      expect(isBusinessHours(earlyMorning, 'UTC')).toBe(false);
    });

    it('should handle late evening correctly', () => {
      // Tuesday 8 PM UTC (after 5 PM)
      const lateEvening = new Date('2024-06-18T20:00:00Z');
      expect(isBusinessHours(lateEvening, 'UTC')).toBe(false);
    });

    it('should respect different timezones', () => {
      // 2 PM in New York, which is 6 PM UTC (business hours in NY)
      const date = new Date('2024-06-18T18:00:00Z');
      expect(isBusinessHours(date, 'America/New_York')).toBe(true);
      expect(isBusinessHours(date, 'UTC')).toBe(false); // 6 PM UTC is after hours
    });
  });

  describe('Edge Cases', () => {
    it('should handle midnight correctly', () => {
      const midnight = new Date('2024-06-15T00:00:00Z');
      expect(getHourInTimezone(midnight, 'UTC')).toBe(0);
      expect(formatHour12(0)).toBe('12a');
    });

    it('should handle noon correctly', () => {
      const noon = new Date('2024-06-15T12:00:00Z');
      expect(getHourInTimezone(noon, 'UTC')).toBe(12);
      expect(formatHour12(12)).toBe('12p');
    });

    it('should handle date boundary crossing', () => {
      // 11 PM UTC on Saturday = Sunday in Tokyo
      const date = new Date('2024-06-15T23:00:00Z');

      const utcDay = getDayInTimezone(date, 'UTC');
      const tokyoDay = getDayInTimezone(date, 'Asia/Tokyo');

      expect(utcDay).not.toBe(tokyoDay);
    });

    it('should handle year boundary', () => {
      const newYearsEve = new Date('2024-12-31T23:59:59Z');
      const newYearsDay = new Date('2025-01-01T00:00:00Z');

      expect(newYearsEve.getUTCFullYear()).toBe(2024);
      expect(newYearsDay.getUTCFullYear()).toBe(2025);
    });

    it('should handle leap year', () => {
      // 2024 is a leap year
      const feb29 = new Date('2024-02-29T12:00:00Z');
      expect(feb29.getUTCDate()).toBe(29);
      expect(feb29.getUTCMonth()).toBe(1); // February
    });

    it('should handle DST transition (spring forward)', () => {
      // March 10, 2024 is DST transition in US
      const beforeDST = new Date('2024-03-10T06:00:00Z'); // 1 AM EST
      const afterDST = new Date('2024-03-10T08:00:00Z'); // 4 AM EDT

      const hourBefore = getHourInTimezone(beforeDST, 'America/New_York');
      const hourAfter = getHourInTimezone(afterDST, 'America/New_York');

      // After DST, same UTC time is one hour later in local time
      expect(hourAfter - hourBefore).toBe(3); // Jumped from 1 AM to 4 AM
    });
  });

  describe('Timestamp Operations', () => {
    it('should handle Unix timestamps', () => {
      const timestamp = 1718452800000; // 2024-06-15T12:00:00Z
      const date = new Date(timestamp);

      expect(date.getUTCFullYear()).toBe(2024);
      expect(date.getUTCMonth()).toBe(5);
      expect(date.getUTCDate()).toBe(15);
      expect(date.getUTCHours()).toBe(12);
    });

    it('should convert between timestamps and dates', () => {
      const original = new Date('2024-06-15T12:00:00Z');
      const timestamp = original.getTime();
      const restored = new Date(timestamp);

      expect(restored.toISOString()).toBe(original.toISOString());
    });

    it('should calculate time differences correctly', () => {
      const date1 = new Date('2024-06-15T12:00:00Z');
      const date2 = new Date('2024-06-15T14:30:00Z');

      const diffMs = date2.getTime() - date1.getTime();
      const diffMinutes = diffMs / 60000;
      const diffHours = diffMs / 3600000;

      expect(diffMinutes).toBe(150);
      expect(diffHours).toBe(2.5);
    });
  });

  describe('Date Comparison', () => {
    it('should compare dates correctly', () => {
      const earlier = new Date('2024-06-15T10:00:00Z');
      const later = new Date('2024-06-15T14:00:00Z');

      expect(earlier < later).toBe(true);
      expect(later > earlier).toBe(true);
      expect(earlier.getTime() < later.getTime()).toBe(true);
    });

    it('should check if dates are the same day', () => {
      const date1 = new Date('2024-06-15T10:00:00Z');
      const date2 = new Date('2024-06-15T22:00:00Z');
      const date3 = new Date('2024-06-16T10:00:00Z');

      const isSameDay = (d1: Date, d2: Date, tz: string) => {
        return formatInTimezone(d1, tz, { dateStyle: 'short' }) ===
               formatInTimezone(d2, tz, { dateStyle: 'short' });
      };

      expect(isSameDay(date1, date2, 'UTC')).toBe(true);
      expect(isSameDay(date1, date3, 'UTC')).toBe(false);
    });
  });

  describe('Mocked Date', () => {
    beforeEach(() => {
      vi.useFakeTimers();
    });

    afterEach(() => {
      vi.useRealTimers();
    });

    it('should mock Date.now() correctly', () => {
      const mockDate = new Date('2024-06-15T12:00:00Z');
      vi.setSystemTime(mockDate);

      expect(Date.now()).toBe(mockDate.getTime());
      expect(new Date().toISOString()).toBe('2024-06-15T12:00:00.000Z');
    });

    it('should allow time advancement', () => {
      const startDate = new Date('2024-06-15T12:00:00Z');
      vi.setSystemTime(startDate);

      const beforeAdvance = Date.now();
      vi.advanceTimersByTime(3600000); // 1 hour
      const afterAdvance = Date.now();

      expect(afterAdvance - beforeAdvance).toBe(3600000);
    });

    it('should handle setInterval with mocked time', () => {
      vi.setSystemTime(new Date('2024-06-15T12:00:00Z'));

      let callCount = 0;
      const interval = setInterval(() => {
        callCount++;
      }, 1000);

      expect(callCount).toBe(0);
      vi.advanceTimersByTime(3000);
      expect(callCount).toBe(3);

      clearInterval(interval);
    });
  });

  describe('Localization', () => {
    it('should format dates in different locales', () => {
      const date = new Date('2024-06-15T12:00:00Z');

      const usFormat = new Intl.DateTimeFormat('en-US', { dateStyle: 'short' }).format(date);
      const ukFormat = new Intl.DateTimeFormat('en-GB', { dateStyle: 'short' }).format(date);

      // US: M/D/YY, UK: DD/MM/YYYY
      expect(usFormat).toMatch(/6\/15\/24/);
      expect(ukFormat).toMatch(/15\/06\/2024/);
    });

    it('should format times in different locales', () => {
      const date = new Date('2024-06-15T14:30:00Z');

      const us12h = new Intl.DateTimeFormat('en-US', {
        timeZone: 'UTC',
        hour: 'numeric',
        minute: '2-digit',
        hour12: true,
      }).format(date);

      const de24h = new Intl.DateTimeFormat('de-DE', {
        timeZone: 'UTC',
        hour: '2-digit',
        minute: '2-digit',
        hour12: false,
      }).format(date);

      expect(us12h).toMatch(/2:30\s*PM/i);
      expect(de24h).toMatch(/14:30/);
    });
  });
});
