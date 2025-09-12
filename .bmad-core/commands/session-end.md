# *session-end Command

## Purpose
Execute comprehensive session ending protocol to ensure clean handoffs and smooth recovery.

## Command Usage
```
*session-end
```

## Execution Sequence

### Phase 1: Work Completion Validation
1. **Check Current Work Status**
   - Verify no partial implementations are left hanging
   - Ensure all files are saved and consistent
   - Validate current task completion state

2. **Code Quality Gates**
   - Run full test suite
   - Check linting and code standards
   - Verify build succeeds
   - Confirm no compilation errors

### Phase 2: State Synchronization
1. **Archon Task Management**
   - Update all task statuses to reflect reality
   - Commit current progress accurately
   - Document any blockers or dependencies
   - Sync priority queue for next session

2. **Git Management**
   - Commit all changes with descriptive messages
   - Ensure clean working directory
   - Document any branch strategy decisions
   - Tag significant milestones if applicable

### Phase 3: Handoff Documentation Generation
1. **Create SESSION-HANDOFF.md** (using session-handoff task)
   - Current project state snapshot
   - Next session priorities
   - Architectural context and decisions
   - Technical notes and issues

2. **Update Project Documentation**
   - Refresh README if needed
   - Update any relevant docs
   - Note configuration changes
   - Document new dependencies

### Phase 4: Final Validation
1. **Health Check**
   - All systems functional
   - No broken dependencies
   - Environment stability confirmed
   - Database/services operational

2. **Recovery Readiness**
   - Handoff documentation complete
   - Next priorities clearly defined
   - Context fully preserved
   - Recovery prompt ready

## Success Output
```
✅ SESSION END PROTOCOL COMPLETE

📋 Handoff Summary:
   - Tasks Completed: [X]
   - Current Task: [task_name] - [status]
   - Next Priority: [next_task]
   
🔍 Validation Status:
   - Tests: ✅ All Passing
   - Build: ✅ Successful  
   - Git: ✅ Clean
   - Archon: ✅ Synchronized
   
📄 Documentation:
   - SESSION-HANDOFF.md: ✅ Generated
   - Recovery Prompt: ✅ Ready
   
🎯 Next Session Ready:
   Use SESSION-RECOVERY-PROMPT.md for immediate restart
   All context preserved and validated
```

## Error Handling
- If tests fail: Fix before ending or document failures
- If git is dirty: Clean up or document intentional state
- If Archon sync fails: Manual sync required
- If documentation incomplete: Block session end until complete

## Integration Notes
- Automatically triggers session-handoff task
- Updates CLAUDE.md if needed
- Prepares recovery prompt for user
- Ensures zero-context-loss transitions