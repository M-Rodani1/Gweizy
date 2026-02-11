export type TransitionMap<State extends string, Event extends string> = {
  [S in State]?: Partial<Record<Event, State>>;
};

export interface StateMachine<State extends string, Event extends string> {
  getState: () => State;
  can: (event: Event) => boolean;
  transition: (event: Event) => State;
}

export const createStateMachine = <State extends string, Event extends string>(
  initial: State,
  transitions: TransitionMap<State, Event>
): StateMachine<State, Event> => {
  let current = initial;

  const getState = () => current;

  const can = (event: Event) => {
    const next = transitions[current]?.[event];
    return typeof next === 'string';
  };

  const transition = (event: Event) => {
    const next = transitions[current]?.[event];
    if (!next) {
      throw new Error(`Invalid transition from ${current} on ${event}`);
    }
    current = next;
    return current;
  };

  return { getState, can, transition };
};
