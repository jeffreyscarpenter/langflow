import React, { forwardRef } from "react";

/**
 * ScrollArea is a reusable scrollable container with consistent styling.
 *
 * Usage:
 * <ScrollArea className="h-64"> ... </ScrollArea>
 */
export interface ScrollAreaProps extends React.HTMLAttributes<HTMLDivElement> {
  children: React.ReactNode;
  className?: string;
  style?: React.CSSProperties;
}

const ScrollArea = forwardRef<HTMLDivElement, ScrollAreaProps>(
  ({ children, className = "", style = {}, ...props }, ref) => {
    return (
      <div
        ref={ref}
        className={`overflow-auto scrollbar-thin scrollbar-thumb-rounded scrollbar-thumb-muted-foreground/50 scrollbar-track-transparent ${className}`}
        style={{ ...style }}
        {...props}
      >
        {children}
      </div>
    );
  }
);

ScrollArea.displayName = "ScrollArea";

export { ScrollArea };